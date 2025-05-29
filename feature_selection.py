import numpy as np 

def feature_selection_per_gait(dat): # for one gait cycle
    # Find the timing (index) of the transition from 0 to 1 in "GAIT_SUBPHASE"
    gait_subphase = dat["GAIT_SUBPHASE"]
    transition_idx = np.where((gait_subphase[:-1] == 0))[0][-1] if sum(gait_subphase[:-1] == 0) > 0 else -1 
    st_sw_phase = transition_idx / 100 
    # print(f"st_sw_phase: {st_sw_phase}")
    # duration of brake time (subphase 4)
    brake_indices = np.where(dat["GAIT_SUBPHASE"] == 4)[0]
    if len(brake_indices) > 0:
        # Assuming 100 Hz sampling rate, so duration = count / 100
        brake_time = len(brake_indices) / 100
    else:
        brake_time = 0

    # duration of toe off assist (subphase 1)
    toe_off_indices = np.where(dat["GAIT_SUBPHASE"] == 1)[0]
    if len(toe_off_indices) > 0:
        # Assuming 100 Hz sampling rate, so duration = count / 100
        toe_off_time = len(toe_off_indices) / 100
    else:
        toe_off_time = 0
    
    # max loadcell values
    max_loadcell = dat["LOADCELL"].max()

    return [
        dat["ACTUATOR_POSITION"][:transition_idx].max(), # max actuator position in st 
        dat["ACTUATOR_POSITION"][:transition_idx].min(), # min actuator position in st
        dat["ACTUATOR_POSITION"][transition_idx:].max(), # max actuator position in sw
        dat["ACTUATOR_POSITION"][transition_idx:].min(), # min actuator position in sw

        dat["KNEE_VELOCITY"][:transition_idx].max(),  # max knee velocity in st
        dat["KNEE_VELOCITY"][:transition_idx].min(),  # min knee velocity in st
        dat["KNEE_VELOCITY"][transition_idx:].max(),  # max knee velocity in sw
        dat["KNEE_VELOCITY"][transition_idx:].min(),  # min knee velocity in sw
        dat["HIP_VELOCITY"][:transition_idx].max(),   # max hip velocity in st
        dat["HIP_VELOCITY"][:transition_idx].min(),   # min hip velocity in st
        dat["HIP_VELOCITY"][transition_idx:].max(),   # max hip velocity in sw
        dat["HIP_VELOCITY"][transition_idx:].min(),   # min hip velocity in sw

        np.argmax(dat["ACTUATOR_POSITION"][:transition_idx]) / 100, # max actuator position phase
        np.argmin(dat["ACTUATOR_POSITION"][:transition_idx]) / 100, # min actuator position phase
        np.argmax(dat["ACTUATOR_POSITION"][transition_idx:]) / 100, # max actuator position phase
        np.argmin(dat["ACTUATOR_POSITION"][transition_idx:]) / 100, # min actuator position phase
        np.argmax(dat["KNEE_VELOCITY"][:transition_idx]) / 100,     # max knee velocity phase in st
        np.argmin(dat["KNEE_VELOCITY"][:transition_idx]) / 100,     # min knee velocity phase in st
        np.argmax(dat["KNEE_VELOCITY"][transition_idx:]) / 100,     # max knee velocity phase in sw
        np.argmin(dat["KNEE_VELOCITY"][transition_idx:]) / 100,     # min knee velocity phase in sw

        np.argmax(dat["HIP_VELOCITY"][:transition_idx]) / 100,      # max hip velocity phase in st
        np.argmin(dat["HIP_VELOCITY"][:transition_idx]) / 100,      # min hip velocity phase in st
        np.argmax(dat["HIP_VELOCITY"][transition_idx:]) / 100,      # max hip velocity phase in sw
        np.argmin(dat["HIP_VELOCITY"][transition_idx:]) / 100,      # min hip velocity phase in sw

        st_sw_phase, 
        brake_time, 
        toe_off_time,
        # max_loadcell,
    ]

def feature_selection(data_list):
    """
    Apply feature selection to a list of data DataFrames.
    Each DataFrame represents a single gait cycle.
    """
    # Apply feature selection to each DataFrame in the list
    return np.array([feature_selection_per_gait(dat) for dat in data_list]),  [
        "max_knee_position_st", "min_knee_position_st",
        "max_knee_position_sw", "min_knee_position_sw",
        "max_knee_velocity_st", "min_knee_velocity_st",
        "max_knee_velocity_sw", "min_knee_velocity_sw",
        "max_hip_velocity_st", "min_hip_velocity_st",
        "max_hip_velocity_sw", "min_hip_velocity_sw",
        "max_knee_position_phase_st", "min_knee_position_phase_st",
        "max_knee_position_phase_sw", "min_knee_position_phase_sw",
        "max_knee_velocity_phase_st", "min_knee_velocity_phase_st",  
        "max_knee_velocity_phase_sw", "min_knee_velocity_phase_sw",
        "max_hip_velocity_phase_st", "min_hip_velocity_phase_st",  
        "max_hip_velocity_phase_sw", "min_hip_velocity_phase_sw",
        "st_sw_phase",
        "brake_time",          
        "toe_off_time",
        # "max_loadcell"
    ]

