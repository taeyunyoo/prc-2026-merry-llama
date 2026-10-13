"""Frozen v52/v50 model settings; v54/v55 rules are separate assets."""
REQUIRED_VERSIONS = {'numpy': '2.2.6', 'pandas': '2.3.3', 'pyarrow': '25.0.0', 'lightgbm': '4.7.0', 'catboost': '1.2.10', 'scikit-learn': '1.7.2'}

WEIGHT = 0.25

MIN_LOCAL_ROWS = 200

LOCAL_TREES = 200

FIXED_TREES = {'lgb': {'EDDF': 648, 'EDDM': 563, 'EGLL': 300, 'EHAM': 458, 'LEBL': 261, 'LEMD': 295, 'LFPG': 253, 'LIRF': 409, 'LSZH': 382, 'LTFM': 526}, 'cb': {'EDDF': 1070, 'EDDM': 1631, 'EGLL': 1539, 'EHAM': 1844, 'LEBL': 1120, 'LEMD': 820, 'LFPG': 1058, 'LIRF': 1747, 'LSZH': 1249, 'LTFM': 1555}, 'cb_deep': {'EDDF': 1398, 'EDDM': 1216, 'EGLL': 2520, 'EHAM': 3509, 'LEBL': 1263, 'LEMD': 2041, 'LFPG': 2047, 'LIRF': 1657, 'LSZH': 1156, 'LTFM': 2492}, 'm9_cb': 127}

MISSING9_LGB_TREES = 76

ID, TIME, TARGET = ('MVT_ID_mvt', 'MVT_TIME_UTC_mvt', 'TAXITIME_SEC_mvt')

MEDIAN = 'stand_runway_median_taxiout_sec'

CATEGORICAL = ['ADEP', 'ADES', 'AIRCRAFT_TYPE', 'FLIGHT_RULE', 'RUNWAY_mvt', 'STAND_mvt', 'MARKET_SEGMENT_flt', 'FLIGHT_TYPE_flt', 'WK_TBL_CAT_flt', 'AIRCRAFT_OPERATOR_flt', 'FLIGHT_mvt', 'CALLSIGN_flt']

LOCAL = ['local_month', 'local_day_of_week', 'local_hour', 'local_weekend_flag', 'local_hour_sin', 'local_hour_cos']

WX = ['wx_temperature_f', 'wx_wind_speed_kt', 'wx_gust_factor_kt', 'wx_gust_reported_flag', 'wx_visibility_sm', 'wx_ceiling_ft', 'wx_no_ceiling_flag', 'wx_metar_available_flag', 'wx_precip_flag', 'wx_snow_ice_flag', 'wx_freezing_precip_flag', 'wx_thunderstorm_flag', 'wx_fog_mist_flag']

DIFF = ['takeoff_minus_schedule_sec', 'aobt3_taxi_proxy_sec', 'takeoff_minus_eobt1_sec', 'takeoff_minus_iobt_sec', 'takeoff_minus_lobt_sec', 'aobt3_minus_schedule_sec', 'eobt1_minus_schedule_sec', 'iobt_minus_schedule_sec', 'lobt_minus_schedule_sec', 'aobt3_minus_eobt1_sec', 'aobt3_minus_iobt_sec', 'aobt3_minus_lobt_sec', 'eobt1_minus_iobt_sec', 'lobt_minus_eobt1_sec', 'lobt_minus_iobt_sec']

COUNTS = ['dep_count_prev_10min', 'dep_count_prev_30min', 'dep_count_prev_60min', 'hs_dep_same_runway_prev_15min']

REF = ['avg_predeparture_delay_sec', 'avg_reference_taxiout_sec', 'avg_additional_taxiout_sec']

BASE_FEATURES = CATEGORICAL + LOCAL + WX + DIFF + COUNTS + REF + ['distance_km', MEDIAN]

AIRPORTS = ['EDDF', 'EDDM', 'EGLL', 'EHAM', 'LEBL', 'LEMD', 'LFPG', 'LIRF', 'LSZH', 'LTFM']

PROXY, SCHEDULE, AOBT = ('aobt3_taxi_proxy_sec', 'takeoff_minus_schedule_sec', 'AOBT_3_flt')

PARAMS = dict(objective='regression', metric='rmse', learning_rate=0.05, num_leaves=31, min_child_samples=100, reg_lambda=1.0, random_state=42, n_jobs=-1, verbosity=-1, deterministic=True, force_col_wise=True, subsample=1.0, colsample_bytree=1.0)

GROUP_FEATURES = {'4B': ['inb_turnaround_sec', 'inb_T_minus_inblock_sec', 'inb_sched_turn_sec', 'inb_match_strict'], '3': ['arr_taxiin_mean_prev30', 'arr_taxiin_n_prev30', 'arr_taxiin_excess_prev30', 'arr_taxiin_trend'], '5': ['arr_rwy_dom', 'arr_rwy_dom_share60', 'arr_rwy_entropy60', 'arr_dir_main60', 'arr_min_since_dir_change', 'arr_same_rwy_prev30', 'arr_same_rwy_share30', 'arr_same_rwy_prev60', 'arr_same_rwy_share60'], 'NXa': ['stand_next_inblock_minus_aobt'], 'B2': ['inb_turn_dev', 'inb_turn_ref_level', 'inb_slack_sec', 'inb2_T_minus_inblock', 'inb_cand_n', 'inb_conf'], 'C2': ['arr_taxiin_routedev_mean30', 'arr_taxiin_routedev_med30', 'arr_taxiin_p80_share30', 'arr_taxiin_since_last'], 'N1': ['inb_sched_record'], 'N2': ['arr_schedrec_share120_T', 'arr_schedrec_n120_T', 'arr_schedrec_share120_A'], 'N5': ['rwy_land_during_taxi', 'rwy_land_during_taxi_per10', 'rwy_land_prev20'], 'N9': ['rwy_since_last_land', 'rwy_gap_med30', 'rwy_gap_std30', 'rwy_gap_max30', 'rwy_land_n30']}

COMMON_GROUPS = ['4B', '3', '5', 'NXa', 'B2', 'C2']

PRESENT_ONLY_GROUPS = ['N1', 'N2', 'N5', 'N9']

ARRIVAL_FEATURES = [f for g in COMMON_GROUPS + PRESENT_ONLY_GROUPS for f in GROUP_FEATURES[g]]

PRESENT_FEATURES = BASE_FEATURES + ARRIVAL_FEATURES

MISSING_FEATURES = BASE_FEATURES + [f for g in COMMON_GROUPS for f in GROUP_FEATURES[g]]

MODEL_CATS = CATEGORICAL + ['arr_rwy_dom']

SPECIAL = 'lirf_special_pred_v29'

LIRF_PARAMS = dict(objective='multiclass', num_class=3, n_estimators=250, learning_rate=0.05, num_leaves=15, min_child_samples=10, reg_lambda=1.0, random_state=42, verbosity=-1, deterministic=True, force_col_wise=True)

DEP_FEATURES = ['tf_dep_rwy_prev_5min', 'tf_dep_rwy_prev_10min', 'tf_dep_rwy_prev_20min',
                'tf_sec_since_prev_dep_rwy', 'tf_sec_to_next_dep_rwy', 'tf_queue_during_taxi_rwy',
                'tf_queue_during_taxi_apt', 'tf_taxiing_at_pushback', 'cf_recent_taxi_apt_30min',
                'cf_recent_taxi_rwy_30min', 'cf_recent_taxi_rwy_60min', 'cf_proxy_minus_recent',
                'cf_recent_vs_usual_rwy', 'cf_n_dep_runways', 'cf_n_arr_runways']

SIDE_FEATURES = ['tf_dep_rwy_prev_5min', 'tf_dep_rwy_prev_10min', 'tf_dep_rwy_prev_20min',
                 'tf_sec_since_prev_dep_rwy', 'tf_sec_to_next_dep_rwy', 'tf_arr_prev_10min', 'tf_arr_prev_20min',
                 'tf_arr_rwy_prev_10min', 'cf_recent_taxi_apt_30min', 'cf_recent_taxi_rwy_30min',
                 'cf_recent_taxi_rwy_60min', 'cf_recent_vs_usual_rwy', 'cf_n_dep_runways', 'cf_n_arr_runways',
                 'ar_taxiing_at_takeoff', 'ar_taxiin_med_30min', 'ar_taxiin_excess_30min', 'ar_delay_mean_60min',
                 'ar_delayed15_share_60min', 'bk_last_arr_gap', 'bk_second_arr_gap', 'bk_last_arr_delay',
                 'bk_n_arr_stand_3h', 'bk_prev_dep_stand_gap', 'bk_next_arr_gap']

ESTIMATE = ['airside_est', 'airside_minus_proxy']

MAIN_FEATURES = PRESENT_FEATURES + DEP_FEATURES + ['cf_config'] + ESTIMATE

MAIN_CATS = MODEL_CATS + ['cf_config']

M9_CB_FEATURES = MISSING_FEATURES + SIDE_FEATURES + ['cf_config', 'airside_est']

CB_PARAMS = dict(loss_function='RMSE', learning_rate=0.08, depth=8, l2_leaf_reg=3.0, random_seed=42, verbose=0,
                 iterations=5000, od_type='Iter', od_wait=200, use_best_model=True)

CB_DEEP = dict(CB_PARAMS, depth=10, learning_rate=0.05, l2_leaf_reg=5.0)

MAIN_MODELS = {'lgb': ('lgb', PARAMS), 'cb': ('cb', CB_PARAMS), 'cb_deep': ('cb', CB_DEEP)}

lirf_THREADS = 4

lirf_OPS = ['dep_count_prev_10min', 'dep_count_prev_30min', 'dep_count_prev_60min', 'hs_dep_same_runway_prev_15min', 'arr_same_rwy_prev30', 'arr_same_rwy_share30', 'arr_same_rwy_prev60', 'arr_same_rwy_share60', 'arr_rwy_dom_share60', 'arr_rwy_entropy60', 'tf_dep_rwy_prev_5min', 'tf_dep_rwy_prev_10min', 'tf_dep_rwy_prev_20min', 'tf_sec_since_prev_dep_rwy', 'tf_arr_prev_10min', 'tf_arr_prev_20min', 'tf_arr_rwy_prev_10min', 'cf_n_dep_runways', 'cf_n_arr_runways', 'ar_taxiin_med_30min', 'ar_delay_mean_60min', 'ar_delayed15_share_60min', 'bk_last_arr_gap', 'bk_second_arr_gap', 'bk_last_arr_delay', 'bk_n_arr_stand_3h', 'bk_prev_dep_stand_gap']

lirf_CORE = ['CALLSIGN_flt', 'FLIGHT_mvt', SCHEDULE, 'local_hour', 'local_month', 'local_day_of_week', 'STAND_mvt', 'AIRCRAFT_TYPE', 'MARKET_SEGMENT_flt', 'ADES', 'distance_km', 'RUNWAY_mvt']

lirf_CATS = ['ICAO', 'AL', 'STAND', 'TYPE', 'SEG', 'ADES', 'RWY']

lirf_REG_PARAMS = dict(objective='regression', learning_rate=0.03, num_leaves=7, max_depth=3, min_child_samples=40, reg_lambda=20.0, random_state=42, verbosity=-1, deterministic=True, force_col_wise=True, n_jobs=lirf_THREADS)

LOCAL_PARAMS = dict(PARAMS, num_leaves=15, min_child_samples=30, reg_lambda=10., n_jobs=4)

AIR_CAT = ['ADEP', 'ADES', 'AIRCRAFT_TYPE', 'FLIGHT_RULE', 'RUNWAY_mvt', 'STAND_mvt', 'WK_TBL_CAT_flt',
           'AIRCRAFT_OPERATOR_flt']

AIR_NUM = LOCAL + WX + [SCHEDULE] + COUNTS + REF + ['distance_km', MEDIAN]

AIR_PARAMS = dict(objective='regression', n_estimators=300, learning_rate=0.1, num_leaves=63,
                  min_child_samples=100, reg_lambda=1.0, colsample_bytree=0.8, random_state=42, verbosity=-1)

AIR_SAMPLE, AIR_CAP = 0.4, 10800

AIR_CAT = ['ADEP', 'ADES', 'AIRCRAFT_TYPE', 'FLIGHT_RULE', 'RUNWAY_mvt', 'STAND_mvt', 'WK_TBL_CAT_flt',
           'AIRCRAFT_OPERATOR_flt']

AIR_NUM = LOCAL + WX + [SCHEDULE] + COUNTS + REF + ['distance_km', MEDIAN]

AIR_PARAMS = dict(objective='regression', n_estimators=300, learning_rate=0.1, num_leaves=63,
                  min_child_samples=100, reg_lambda=1.0, colsample_bytree=0.8, random_state=42, verbosity=-1)

AIR_SAMPLE, AIR_CAP = 0.4, 10800

AIR_CAT = ['ADEP', 'ADES', 'AIRCRAFT_TYPE', 'FLIGHT_RULE', 'RUNWAY_mvt', 'STAND_mvt', 'WK_TBL_CAT_flt',
           'AIRCRAFT_OPERATOR_flt']

AIR_NUM = LOCAL + WX + [SCHEDULE] + COUNTS + REF + ['distance_km', MEDIAN]

AIR_PARAMS = dict(objective='regression', n_estimators=300, learning_rate=0.1, num_leaves=63,
                  min_child_samples=100, reg_lambda=1.0, colsample_bytree=0.8, random_state=42, verbosity=-1)

AIR_SAMPLE, AIR_CAP = 0.4, 10800
