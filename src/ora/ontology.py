from __future__ import annotations

import collections
import os
from typing import Any, Dict, Iterable, Iterator, Optional, Set

import meds_reader
import polars as pl


def _get_all_codes_map(subjects: Iterator[meds_reader.Subject]) -> Set[str]:
    result = set()
    for subject in subjects:
        for event in subject.events:
            result.add(event.code)
    return result


class Ontology:
    def __init__(self, athena_path: str, code_metadata_path: Optional[str] = None):
        """Create an Ontology from an Athena download and an optional meds Code Metadata structure.

        NOTE: This is an expensive operation.
        It is recommended to create an ontology once and then save/load it as necessary.
        """
        # Load from code metadata
        self.description_map: Dict[str, str] = {}
        self.parents_map: Dict[str, Set[str]] = collections.defaultdict(set)

        # Load from the athena path ...
        concept = pl.scan_csv(os.path.join(athena_path, "CONCEPT.csv"), separator="\t", infer_schema_length=0)
        code_col = pl.col("vocabulary_id") + "/" + pl.col("concept_code")
        description_col = pl.col("concept_name")
        concept_id_col = pl.col("concept_id").cast(pl.Int64)

        processed_concepts = (
            concept.select(code_col, concept_id_col, description_col, pl.col("standard_concept").is_null())
            .collect()
            .rows()
        )

        concept_id_to_code_map = {}

        non_standard_concepts = set()

        for code, concept_id, description, is_non_standard in processed_concepts:
            concept_id_to_code_map[concept_id] = code

            # We don't want to override code metadata
            if code not in self.description_map:
                self.description_map[code] = description

            if is_non_standard:
                non_standard_concepts.add(concept_id)

        relationship = pl.scan_csv(
            os.path.join(athena_path, "CONCEPT_RELATIONSHIP.csv"), separator="\t", infer_schema_length=0
        )
        relationship_id = pl.col("relationship_id")
        # only keep 'map to' relationship
        relationship = relationship.filter(
            relationship_id == "Maps to", pl.col("concept_id_1") != pl.col("concept_id_2")
        )
        for concept_id_1, concept_id_2 in (
            relationship.select(pl.col("concept_id_1").cast(pl.Int64), pl.col("concept_id_2").cast(pl.Int64))
            .collect()
            .rows()
        ):
            # for nonstandard concept, “Maps to” is treated as a parent link. if code1 maps to code2, self.parents_map[code1] = code2
            if concept_id_1 in non_standard_concepts:
                self.parents_map[concept_id_to_code_map[concept_id_1]].add(concept_id_to_code_map[concept_id_2])

        # update parent relationship with CONCEPT_ANCESTOR.csv
        ancestor = pl.scan_csv(os.path.join(athena_path, "CONCEPT_ANCESTOR.csv"), separator="\t", infer_schema_length=0)
        ancestor = ancestor.filter(pl.col("min_levels_of_separation") == "1")
        for concept_id, parent_concept_id in (
            ancestor.select(
                pl.col("descendant_concept_id").cast(pl.Int64), pl.col("ancestor_concept_id").cast(pl.Int64)
            )
            .collect()
            .rows()
        ):
            self.parents_map[concept_id_to_code_map[concept_id]].add(concept_id_to_code_map[parent_concept_id])

        if code_metadata_path is not None:
            code_metadata = pl.scan_parquet(code_metadata_path)
            print(code_metadata)
            # code_metadat_items = (
            #     code_metadata.select(pl.col("code"), pl.col("description"), pl.col("parent_codes")).collect().to_dicts()
            # )
            code_metadat_items = (
                code_metadata.select(pl.col("code")).collect().to_dicts()
            )

            # Have to add after OMOP to overwrite ...
            for code_info in code_metadat_items:
                code = code_info.get("code")
                if code is not None:
                    # If description present → overwrite description_map[code]
                    # If parent_codes present → overwrite parents_map[code] with that set (filtered for non-None)
                    if code_info.get("description") is not None:
                        self.description_map[code] = code_info["description"]
                    if code_info.get("parent_codes") is not None:
                        self.parents_map[code] = set(i for i in code_info["parent_codes"] if i is not None)

        # create
        self.children_map = collections.defaultdict(set)
        for code, parents in self.parents_map.items():
            for parent in parents:
                self.children_map[parent].add(code)

        self.all_parents_map: Dict[str, Set[str]] = {}
        self.all_children_map: Dict[str, Set[str]] = {}

    def prune_to_dataset(
        self,
        data_pool: meds_reader.SubjectDatabase,
        prune_all_descriptions: bool = False,
        remove_ontologies: Set[str] = set(),
    ) -> None:
        valid_codes = set()
        for chunk_codes in data_pool.map(_get_all_codes_map):
            valid_codes |= chunk_codes

        # prune_all_descriptions is set to true in motor
        if prune_all_descriptions:
            self.description_map = {}

        all_parents = set()

        # every valid code + all ontology ancestors reachable via parents_map
        for code in valid_codes:
            all_parents |= self.get_all_parents(code)

        # keep a code if either:
        # it’s directly present in your dataset, always keep
        # OR it’s an ancestor of some dataset code, but only if its vocabulary is not removed

        def is_valid(code):
            ontology = code.split("/")[0]
            return (code in valid_codes) or ((ontology not in remove_ontologies) and (code in all_parents))

        codes = self.children_map.keys() | self.parents_map.keys() | self.description_map.keys()
        for code in codes:
            m: Any
            if is_valid(code):
                # update children_map and parents_map to only include valid elements
                for m in (self.children_map, self.parents_map):
                    m[code] = {a for a in m[code] if is_valid(a)}
            else:
                # delete code from all dictionaries
                for m in (self.children_map, self.parents_map, self.description_map):
                    if code in m:
                        del m[code]

        self.all_parents_map = {}
        self.all_children_map = {}

        # Prime the pump
        for code in self.children_map.keys() | self.parents_map.keys():
            self.get_all_parents(code)

    def get_description(self, code: str) -> Optional[str]:
        """Get a description of a code."""
        return self.description_map.get(code)

    def get_children(self, code: str) -> Iterable[str]:
        """Get the children for a given code."""
        return self.children_map.get(code, set())

    def get_parents(self, code: str) -> Iterable[str]:
        """Get the parents for a given code."""
        return self.parents_map.get(code, set())

    def get_all_children(self, code: str) -> Set[str]:
        """Get all children, including through the ontology."""
        if code not in self.all_children_map:
            result = {code}
            for child in self.children_map.get(code, set()):
                result |= self.get_all_children(child)
            self.all_children_map[code] = result
        return self.all_children_map[code]

    def get_all_children_for_codes(self, codes: Set[str]) -> Set[str]:
        result = set()
        for code in codes:
            result |= self.get_all_children(code)
        return result

    # recurviseluy get all grandparents of current code
    def get_all_parents(self, code: str) -> Set[str]:
        """Get all parents, including through the ontology."""
        if code not in self.all_parents_map:
            result = {code}
            for parent in self.parents_map.get(code, set()):
                result |= self.get_all_parents(parent)
            self.all_parents_map[code] = result

        return self.all_parents_map[code]

    def get_all_parents_for_codes(self, codes: Set[str]) -> Set[str]:
        result = set()
        for code in codes:
            result |= self.get_all_parents(code)
        return result
