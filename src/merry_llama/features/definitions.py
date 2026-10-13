"""Frozen feature names from the v42/v50/v52 source lineage."""
from .core import ID, TIME, TIMEZONES

AOBT = 'AOBT_3_flt'
TARGET = 'TAXITIME_SEC_mvt'
PROXY = 'aobt3_taxi_proxy_sec'
MEDIAN = 'stand_runway_median_taxiout_sec'
AIRPORTS = list(TIMEZONES)
GROUP_FEATURES = {'4B': ['inb_turnaround_sec', 'inb_T_minus_inblock_sec', 'inb_sched_turn_sec', 'inb_match_strict'], '3': ['arr_taxiin_mean_prev30', 'arr_taxiin_n_prev30', 'arr_taxiin_excess_prev30', 'arr_taxiin_trend'], '5': ['arr_rwy_dom', 'arr_rwy_dom_share60', 'arr_rwy_entropy60', 'arr_dir_main60', 'arr_min_since_dir_change', 'arr_same_rwy_prev30', 'arr_same_rwy_share30', 'arr_same_rwy_prev60', 'arr_same_rwy_share60'], 'NXa': ['stand_next_inblock_minus_aobt'], 'B2': ['inb_turn_dev', 'inb_turn_ref_level', 'inb_slack_sec', 'inb2_T_minus_inblock', 'inb_cand_n', 'inb_conf'], 'C2': ['arr_taxiin_routedev_mean30', 'arr_taxiin_routedev_med30', 'arr_taxiin_p80_share30', 'arr_taxiin_since_last'], 'N1': ['inb_sched_record'], 'N2': ['arr_schedrec_share120_T', 'arr_schedrec_n120_T', 'arr_schedrec_share120_A'], 'N5': ['rwy_land_during_taxi', 'rwy_land_during_taxi_per10', 'rwy_land_prev20'], 'N9': ['rwy_since_last_land', 'rwy_gap_med30', 'rwy_gap_std30', 'rwy_gap_max30', 'rwy_land_n30']}

COMMON_GROUPS = ['4B', '3', '5', 'NXa', 'B2', 'C2']

PRESENT_ONLY_GROUPS = ['N1', 'N2', 'N5', 'N9']

DEP_FEATURES = ['tf_dep_rwy_prev_5min', 'tf_dep_rwy_prev_10min', 'tf_dep_rwy_prev_20min', 'tf_sec_since_prev_dep_rwy', 'tf_sec_to_next_dep_rwy', 'tf_queue_during_taxi_rwy', 'tf_queue_during_taxi_apt', 'tf_taxiing_at_pushback', 'cf_recent_taxi_apt_30min', 'cf_recent_taxi_rwy_30min', 'cf_recent_taxi_rwy_60min', 'cf_proxy_minus_recent', 'cf_recent_vs_usual_rwy', 'cf_n_dep_runways', 'cf_n_arr_runways']

SIDE_FEATURES = ['tf_dep_rwy_prev_5min', 'tf_dep_rwy_prev_10min', 'tf_dep_rwy_prev_20min', 'tf_sec_since_prev_dep_rwy', 'tf_sec_to_next_dep_rwy', 'tf_arr_prev_10min', 'tf_arr_prev_20min', 'tf_arr_rwy_prev_10min', 'cf_recent_taxi_apt_30min', 'cf_recent_taxi_rwy_30min', 'cf_recent_taxi_rwy_60min', 'cf_recent_vs_usual_rwy', 'cf_n_dep_runways', 'cf_n_arr_runways', 'ar_taxiing_at_takeoff', 'ar_taxiin_med_30min', 'ar_taxiin_excess_30min', 'ar_delay_mean_60min', 'ar_delayed15_share_60min', 'bk_last_arr_gap', 'bk_second_arr_gap', 'bk_last_arr_delay', 'bk_n_arr_stand_3h', 'bk_prev_dep_stand_gap', 'bk_next_arr_gap']

ARRIVAL_FEATURES = [f for g in COMMON_GROUPS + PRESENT_ONLY_GROUPS for f in GROUP_FEATURES[g]]
MOVEMENT_FEATURES = list(dict.fromkeys(DEP_FEATURES + SIDE_FEATURES + ['cf_config']))
# This difference depends on the OOF/fitted route median and is added by training.
MOVEMENT_FEATURES.remove('cf_recent_vs_usual_rwy')
ADDED = ARRIVAL_FEATURES + MOVEMENT_FEATURES
CATEGORICAL_ADDED = ['arr_rwy_dom', 'cf_config']
DEFERRED = [MEDIAN, 'cf_recent_vs_usual_rwy', 'airside_est', 'airside_minus_proxy']
assert len(ARRIVAL_FEATURES) == 40
assert len(ADDED) == len(set(ADDED))
