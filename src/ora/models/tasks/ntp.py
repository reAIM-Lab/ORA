from __future__ import annotations

import abc
import collections
import datetime
import functools
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

import meds
import meds_reader
import numpy as np
import scipy.sparse
import torch
import warnings

import ora.models.config
import ora.models.tokenizer
import ora.ontology
import ora.pat_utils
import ora.stat_utils
import random


class Task(abc.ABC):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def get_task_config(self) -> ora.models.config.FEMRTaskConfig: ...

    @abc.abstractmethod
    def start_batch(self) -> None: ...

    @abc.abstractmethod
    def start_subject(self, subject: meds_reader.Subject, ontology: Optional[ora.ontology.Ontology]) -> None: ...

    @abc.abstractmethod
    def add_subject_labels(self, subject_label_offsets: List[int]) -> None: ...

    @abc.abstractmethod
    def needs_exact(self) -> bool: ...

    @abc.abstractmethod
    def get_sampled_labels(self, length: int) -> int:
        return length

    @abc.abstractmethod
    def add_event(
            self,
            current_date: datetime.datetime,
            next_date: Optional[datetime.datetime],
            next_features: Optional[Sequence[int]],
    ) -> int: ...

    @abc.abstractmethod
    def get_batch_data(self) -> Mapping[str, np.ndarray]: ...

    def cleanup(self, batch: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
        return batch


class LabeledSubjectTask(Task):

    def __init__(self, labels: Sequence[meds.Label], observation_window: Optional[int] = None):
        super().__init__()
        self.label_map: Mapping[int, Any] = collections.defaultdict(list)
        for label in labels:
            self.label_map[label["subject_id"]].append(label)

        for k, v in self.label_map.items():
            v.sort(key=lambda a: a["prediction_time"])

        if observation_window is not None:
            assert observation_window > 0, "the feature extract observation window must be greater than 0 or None"
        self.observation_window = observation_window

    def get_task_config(self) -> ora.models.config.FEMRTaskConfig:
        return ora.models.config.FEMRTaskConfig(task_type="labeled_subjects")

    def start_subject(self, subject: meds_reader.Subject, _ontology: Optional[ora.ontology.Ontology]) -> None:
        self.current_labels = self.label_map[subject.subject_id]
        self.current_label_index = 0

    def needs_exact(self) -> bool:
        return True

    def start_batch(self) -> None:
        """LabeledSubjectTask currently has no per label state."""
        pass

    def add_subject_labels(self, _subject_label_offsets: List[int]) -> None:
        """As there is no per label state, this is ignored"""
        pass

    def add_event(
            self,
            current_date: datetime.datetime,
            next_date: Optional[datetime.datetime],
            next_features: Optional[Sequence[int]] = None,
            actually_add: Optional[bool] = True,
    ) -> int:
        has_label = False

        while True:
            if self.current_label_index == len(self.current_labels):
                break

            current_label = self.current_labels[self.current_label_index]

            if self.observation_window is not None:
                observation_start_time = (
                        current_label["prediction_time"] - datetime.timedelta(days=self.observation_window)
                )
                is_valid = observation_start_time <= current_date <= current_label["prediction_time"]
                next_valid = (
                        next_date is not None and
                        observation_start_time <=next_date <= current_label["prediction_time"]
                )
            else:
                is_valid = current_date <= current_label["prediction_time"]
                next_valid = next_date is not None and next_date <= current_label["prediction_time"]

            if next_valid:
                # Next one is valid, so break early to give it a chance next time
                break

            if is_valid:
                has_label = True
                self.current_label_index += 1
            else:
                # The next label isn't valid, so we have to break here
                break

        if has_label:
            return 1
        else:
            return 0

    def get_batch_data(self) -> Mapping[str, np.ndarray]:
        return {}

    def get_sampled_labels(self, length: int) -> int:
        return length


class CLMBRTask(Task):
    def __init__(self, clmbr_vocab_size: int):
        self.clmbr_vocab_size = clmbr_vocab_size

    def get_task_config(self) -> ora.models.config.FEMRTaskConfig:
        return ora.models.config.FEMRTaskConfig(
            task_type="clmbr", task_kwargs=dict(clmbr_vocab_size=self.clmbr_vocab_size)
        )

    def start_subject(self, _subject: meds_reader.Subject, _ontology: Optional[ora.ontology.Ontology]) -> None:
        self.per_subject_batch_labels: List[int] = []

    def needs_exact(self) -> bool:
        return False

    def start_batch(self) -> None:
        self.batch_labels: List[int] = []

    def add_subject_labels(self, subject_label_offsets: List[int]) -> None:
        self.batch_labels.extend([self.per_subject_batch_labels[i] for i in subject_label_offsets])

    def add_event(
            self,
            current_date: datetime.datetime,
            next_date: Optional[datetime.datetime],
            next_features: Optional[Sequence[int]] = None,
            actually_add: Optional[bool] = True,
    ) -> int:
        if next_features is None:
            return 0

        if len(next_features) != 1:
            raise RuntimeError("Only supports one for right now")

        next_feature = next_features[0]

        if next_feature >= self.clmbr_vocab_size:
            return 0

        self.per_subject_batch_labels.append(next_feature)

        return 1

    def get_batch_data(self) -> Mapping[str, np.ndarray]:
        return {"labels": np.array(self.batch_labels, dtype=np.int32)}

    def get_sampled_labels(self, length: int) -> int:
        return length


