def build_all_cohorts(
    subject_iter,
    code_dict: Dict[str, str],
    visits_dict: Dict[int, np.ndarray]
) -> Dict[str, List[Dict]]:
    """
    Build cohorts for all lab codes in a single pass through the data.
    
    This is more efficient than multiple passes because:
    1. Subject iteration is expensive (disk I/O)
    2. Visit lookup is O(1) per subject
    3. We can extract events for all codes in one pass
    """
    # Initialize result containers
    results = {lab_name: [] for lab_name in code_dict.values()}
    
    # Create reverse mapping: lab_name -> code
    name_to_code = {v: k for k, v in code_dict.items()}
    target_codes = set(code_dict.keys())
    
    # Statistics per lab
    stats = {lab: {'subjects_with_events': 0, 'total_predictions': 0, 'valid_predictions': 0} 
             for lab in code_dict.values()}
    
    total_subjects = 0
    subjects_with_visits = 0
    
    for subject in subject_iter:
        total_subjects += 1
        subject_id = subject.subject_id
        
        if subject_id not in visits_dict:
            continue
        
        subjects_with_visits += 1
        visit_array = visits_dict[subject_id]
        
        # Collect events per lab code
        events_by_lab = {lab: ([], []) for lab in code_dict.values()}
        
        for event in subject.events:
            if event.code in target_codes and event.time is not None:
                try:
                    val = event.numeric_value
                    if val is not None:
                        float_val = float(val)
                        ts = int(event.time.timestamp())
                        lab_name = code_dict[event.code]
                        events_by_lab[lab_name][0].append(ts)
                        events_by_lab[lab_name][1].append(float_val)
                except (TypeError, ValueError):
                    continue
        
        # Process each lab code
        prediction_times = np.unique(visit_array[:, 1])
        
        for lab_name in code_dict.values():
            timestamps_list, values_list = events_by_lab[lab_name]
            
            if not timestamps_list:
                continue
            
            stats[lab_name]['subjects_with_events'] += 1
            
            event_timestamps = np.array(timestamps_list, dtype=np.int64)
            event_values = np.array(values_list, dtype=np.float64)
            
            for pred_ts in prediction_times:
                stats[lab_name]['total_predictions'] += 1
                
                window_start = pred_ts + WINDOW_START_SECS
                window_end = pred_ts + WINDOW_END_SECS
                
                left_idx = np.searchsorted(event_timestamps, window_start, side='left')
                right_idx = np.searchsorted(event_timestamps, window_end, side='right')
                
                for idx in range(left_idx, right_idx):
                    event_ts = event_timestamps[idx]
                    
                    if is_event_in_any_visit_vectorized(event_ts, visit_array):
                        stats[lab_name]['valid_predictions'] += 1
                        
                        pred_datetime = datetime.fromtimestamp(pred_ts)
                        event_datetime = datetime.fromtimestamp(event_ts)
                        
                        results[lab_name].append({
                            'subject_id': subject_id,
                            'prediction_time': pred_datetime,
                            'event_time': event_datetime,
                            'regression_value': event_values[idx]
                        })
                        break