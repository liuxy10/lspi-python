import numpy as np
import os
from lspi.sample import Sample

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from plot_traj_sagittal import plot_leg_sagittal_plane
import statsmodels.api as sm
# import pykeyboard # Ensure this library is installed and compatible with your environment
from pynput import keyboard

def vis_hist(his, skip = 1, names = None):
    # visualize all histories as subplots on the same figure


    fig, axes = plt.subplots(len(his), 1, figsize=(6, 1.5 * len(his)), sharex=True)
    if len(his) == 1:
        axes = [axes]
    for idx, (k, v) in enumerate(his.items()):
        v = np.array(v)
        ax = axes[idx]
        if v.ndim == 1:
            ax.plot(np.arange(0, v.shape[0],step = skip), v[::skip], marker='o', alpha=0.5)
        else:
            for i in range(v.shape[1]):
                ax.plot(np.arange(0, v.shape[0],step = skip), v[::skip, i], marker='o', alpha=0.5, label=f"{k}_{i}" if names is None or k not in names.keys() else f"{names[k][i]}")
            ax.legend()
        ax.set_title(f"History of {k} over iterations")
        ax.set_ylabel(k)
    axes[-1].set_xlabel("Iteration")
    plt.tight_layout()
    plt.show()

def feature_regression_analysis(state_mean, state_std, state_names, params, param_names=None, print_results=True, vis = True):
    """
    Perform weighted least squares regression of each feature mean against params.

    Args:
        state_mean (np.ndarray): Mean values for each feature (n_param_sets x n_features).
        state_std (np.ndarray): Std values for each feature (n_param_sets x n_features).
        state_names (np.ndarray or list): Names of features (n_features,).
        params (np.ndarray): Parameter values (n_param_sets x n_params).
        param_names (list or None): Optional, names of parameters.
        print_results (bool): If True, print regression results.

    Returns:
        results_df (pd.DataFrame): DataFrame with regression stats for each feature.
    """
    

    # create a dictionary to hold feature means, stds, and weights
    feature_dict = {}
    
    for i in range(len(state_names)): # iterate over features
        feature_name = state_names[i]
        means, sds = [], []
        for j in range(len(state_mean)): # iterate over parameter sets
            means.append(float(state_mean[j][i]))
            sds.append(float(state_std[j][i]) if float(state_std[j][i]) != 0 else 1e-6)
        feature_dict[feature_name] = {'means': means, 
                                      'sds': sds, 
                                      "weights": 1}  # initialize weights to 1


    if param_names is None:
        param_names = [f'param{i+1}' for i in range(params.shape[1])]
    params_df = pd.DataFrame(params, columns=param_names)
    
    results = []
 
    for feature, data in feature_dict.items():
        means = np.array(data['means'])
        sds = np.array(data['sds'])
        # weights = np.array(data['weights'])

        wls = False# not making sense
        if wls:
            # Calculate variances
            global_var = np.var(means, ddof=1)  # Sample variance of means
            within_vars = sds**2
            
            # Handle zero variances
            within_vars = np.where(within_vars == 0, 1e-6, within_vars)
            
            # Compute standardized weights
            weights = global_var / within_vars
            
            # Normalize weights
            weights /= weights.mean()  # Optional but often recommended
    
        else:
            # Use standard deviation as weights
            weights = 1


        X = sm.add_constant(params_df) # params are predictors
        model = sm.WLS(means, X, weights=weights).fit()

        results.append({
            'feature': feature,
            'F': model.fvalue,
            'p': model.f_pvalue,
            'R²': model.rsquared,
            'params': [f"{p:.4f}" for p in model.params],  # coefficients rounded to 4 digits
            # 'weights': weights,
        })

    results_df = pd.DataFrame(results).sort_values('p')
    pd.options.display.float_format = '{:.4f}'.format

    if print_results:
        print("Weighted Regression Results (sorted by p-value):")
        print(results_df[['feature', 'F', 'p', 'R²', 'params']].to_string(index=False))
        print("\nSignificant Features (p < 0.05):")
        print(results_df[results_df.p < 0.05][['feature', 'F', 'p',  'R²', 'params']].to_string(index=False))
        
        verbose = False
        if verbose: 
            # also print the regression results for each significant feature
            print("\nDetailed Regression weights for Significant Features:")
            for index, row in results_df[results_df.p < 0.05].iterrows():
                feature = row['feature']
                means = np.array(feature_dict[feature]['means'])
                sds = np.array(feature_dict[feature]['sds'])
                if wls:
                    # Calculate variances
                    global_var = np.var(means, ddof=1)  # Sample variance of means
                    within_vars = sds**2
                    
                    # Handle zero variances
                    within_vars = np.where(within_vars == 0, 1e-6, within_vars)
                    
                    # Compute standardized weights
                    weights = global_var / within_vars
                    
                    # Normalize weights
                    weights /= weights.mean()  # Optional but often recommended
                else:
                    weights = feature_dict[feature]['weights']
                X = sm.add_constant(params_df)
                model = sm.WLS(means, X, weights=weights).fit()
                print(f"\nFeature: {feature}")
                print(model.summary())

    if vis:
            # R squared values visualization
        plt.figure(figsize=(10, 8)) # Increased figure height for better label visibility
        plt.barh(results_df['feature'], results_df['R²'], color='skyblue')
        plt.xlabel('R² Value')
        plt.title('Feature Regression R² Values')
        plt.grid(axis='x')
        plt.gca().invert_yaxis() # To match the order of print output (highest R² at top)
        plt.tight_layout() # Adjust layout to make room for feature names
        plt.show()

        # p-values visualization
        plt.figure(figsize=(10, 8)) # Increased figure height
        plt.barh(results_df['feature'], results_df['p'], color='lightgreen')
        plt.xlabel('p-value')
        plt.title('Feature Regression p-values')
        plt.axvline(x=0.05, color='red', linestyle='--', label='p = 0.05 (Significance Threshold)')
        plt.legend()
        plt.grid(axis='x')
        plt.gca().invert_yaxis() # To match the order of print output
        plt.tight_layout() # Adjust layout
        plt.show()

        # coefficients visualization via heatmap
        plt.figure(figsize=(12, 8))
        coeffs = np.array([[float(val) for val in row['params']] for _, row in results_df.iterrows()])
        sns.heatmap(coeffs, annot=True, fmt=".4f", cmap='coolwarm', 
                    xticklabels=np.concatenate([["bias"], param_names]), yticklabels=results_df['feature'], cbar_kws={'label': 'Coefficient Value'})
        plt.title('Feature Coefficients Heatmap with normalized I/O')
        plt.xlabel('Parameters')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.show()
    return results_df, model


def visualize_states(states, ids = [0,1,2], names = None):
    if len(ids) != 3:
        raise ValueError("Please provide exactly three indices for 3D visualization.")
    # visualize the first three features of the states in 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Use a colormap to get distinct colors for each parameter set
    colors = plt.cm.get_cmap('viridis', len(states))

    for i, state in enumerate(states):
        ax.scatter(state[:, ids[0]], state[:, ids[1]], state[:, ids[2]], color=colors(i), label=f'param set {i}', alpha=0.5)
    if names is not None:
        ax.set_xlabel(f'feature {ids[0]} ({names[ids[0]]})')
        ax.set_ylabel(f'feature {ids[1]} ({names[ids[1]]})')
        ax.set_zlabel(f'feature {ids[2]} ({names[ids[2]]})')
    else:
        ax.set_xlabel(f'feature {ids[0]}')
        ax.set_ylabel(f'feature {ids[1]}')
        ax.set_zlabel(f'feature {ids[2]}')
    ax.set_title('3D Visualization of States')
    ax.legend()
    
    plt.show()



def visualize_gait_cycle(cfg, file, data_list):
    plot_data(file, cfg, data_list,["LOADCELL"], ylim=[-30, 180])
    plot_data(file, cfg, data_list,["ACTUATOR_POSITION"], ylim=[-60,130])
    plot_data(file, cfg, data_list,["SHANK_ANGLE"], ylim=[-60,110])
    plot_data(file, cfg, data_list,["GAIT_PHASE"], ylim=[-0.1,1.1])
    plot_data(file, cfg, data_list,["GAIT_SUBPHASE"], ylim=[-0.1,4.1])
    plot_data(file, cfg, data_list,[f"LINEAR_ACC_{i}_LOCAL" for i in ["X","Y","Z"]], ylim=[-30, 30], mark_max_min=False)
    plot_data(file, cfg, data_list,[f"GRAVITY_VECTOR_{i}" for i in ["X","Y","Z"]], ylim=[-10, 20], mark_max_min=False)
    plot_data(file, cfg, data_list, ["ACTUATOR_SETPOINT","TORQUE_ESTIMATE"], ylim=[-110, 110] )
    plot_data(file, cfg, data_list, ["HIP_VELOCITY", "KNEE_VELOCITY"], ylim=[-60, 60])

def plot_data(file, cfg, data_list, names, ylim, 
              plot_mean_std = True, 
              mark_max_min = True,
              show = False):
    fig = plt.figure(figsize=(10, 5))
    
    for name in names:
        for i in range(len(data_list)):    
            plt.plot(np.arange(0,1,0.01), data_list[i][name], linewidth = .7) #color = "grey")# s=1.5, c="grey")#, label=f"gait cycle {i+1}", alpha=0.2)
        
        # if mark_max_min_individual:
        #     for i in range(len(data_list)):
        #         max_idx = np.argmax(data_list[i][name])
        #         min_idx = np.argmin(data_list[i][name])
        #         plt.scatter(max_idx/100, data_list[i][name].max(), s=10, c="red", marker="o")
        #         plt.scatter(min_idx/100, data_list[i][name].min(), s=10, c="blue", marker="x")
        
        if plot_mean_std:
            mean = np.mean(np.array([data[name] for data in data_list]), axis=0)
            std = np.std(np.array([data[name] for data in data_list]), axis=0)
            plt.plot(np.arange(0,1,0.01), mean, label=f"mean {name.lower()}", color = "black", linewidth = 2)
            plt.fill_between(np.arange(0,1,0.01), mean-3*std, mean+3*std, alpha=0.2, color= "black")
            # mean_phase_change =  mean_st_sw_phase(data_list, loadcell_threshold = cfg["loadcell_threshold"])
            phase_change = np.array([np.where(data["GAIT_PHASE"])[0][0] for data in data_list])
            mean_phase_change = phase_change.mean()
            # mean_phase_change = np.min([np.where(data["GAIT_SUBPHASE"]> 0)[0][0] for data in data_list])
            plt.axvline(mean_phase_change / 100, color='green', linestyle='--')
            plt.text(mean_phase_change / 100, ylim[1] - 0.05 * (ylim[1] - ylim[0]), 
                     f"pc:{mean_phase_change/100:.2f}", fontsize=10, color="green")
            # plot the phase change point
            plt.scatter(phase_change/100, [data_list[j][name].iloc[int(phase_change[j])] for j in range(len(data_list))], s=15, c="green", marker="o")
            if mark_max_min:
                
                min_idx_before = np.nanargmin(mean[:int(mean_phase_change)])
                max_idx_before = np.nanargmax(mean[:int(mean_phase_change)]) # restrict max to happened before min
                max_idx_after = np.nanargmax(mean[int(mean_phase_change):]) + int(mean_phase_change)
                min_idx_after = np.nanargmin(mean[int(mean_phase_change):]) + int(mean_phase_change)

                
                plt.scatter(max_idx_before/100, mean[max_idx_before], s=15, c="red", marker="o")
                plt.scatter(min_idx_before/100, mean[min_idx_before], s=15, c="blue", marker="x")
                plt.axvline(max_idx_before / 100, color='red', linestyle='--')
                plt.axvline(min_idx_before / 100, color='blue', linestyle='--')
                plt.text(max_idx_before/100, mean[max_idx_before] + 0.05 * (ylim[1] - ylim[0]), 
                         f"max: {mean[max_idx_before]:.2f} at {max_idx_before/100:.2f}", fontsize=10, color="red")
                plt.text(min_idx_before/100, mean[min_idx_before] - 0.05 * (ylim[1] - ylim[0]), 
                         f"min: {mean[min_idx_before]:.2f} at {min_idx_before/100:.2f}", fontsize=10, color="blue")
                plt.axvline(max_idx_after / 100, color='red', linestyle='--')
                plt.axvline(min_idx_after/ 100, color='blue', linestyle='--')
                plt.scatter(max_idx_after/100, mean[max_idx_after], s=15, c="red", marker="o")
                plt.scatter(min_idx_after/100, mean[min_idx_after], s=15, c="blue", marker="x")
                plt.text(max_idx_after/100, mean[max_idx_after] + 0.05 * (ylim[1] - ylim[0]), 
                         f"max: {mean[max_idx_after]:.2f} at {max_idx_after/100:.2f}", fontsize=10, color="red")
                plt.text(min_idx_after/100, mean[min_idx_after] - 0.05 * (ylim[1] - ylim[0]), 
                         f"min: {mean[min_idx_after]:.2f} at {min_idx_after/100:.2f}", fontsize=10, color="blue")
                
    plt.ylim(ylim[0], ylim[1])
    plt.legend()
    plt.grid()
    plt.xlabel("phase")
    plt.ylabel("/".join(names).lower())
    os.makedirs("plot", exist_ok=True)
    plt.title(f"{file[:-4]}")
    plt.savefig(f"plot/{'_'.join(names)}_{file[:-4]}.png".lower())
    if show:
        plt.show()
    plt.close()

# def mean_st_sw_phase(data_list, loadcell_threshold = 0.5):
#     return np.mean([np.where(np.where(data_list[i]["LOADCELL"] < loadcell_threshold, 1, 0))[0][0] for i in range(len(data_list))])


def quad_cost(ssa_samples, w_s, n_action, n_state, w_a):
    rew = 1/2 * np.sum(ssa_samples[:,:n_state] **2, axis=1) * w_s + 1/2 * np.sum(ssa_samples[:,-n_action:]**2, axis=1) * w_a
    rew = rew.reshape(-1, 1)
    return rew

def add_quadratic_cost_stack(ssa_samples, n_action, w_s = 0.8, cost_func = None):
    """
    Add a quadratic reward stack to the samples.
    The reward is calculated as:
    reward = -1/2 * ||s - s_target||^2 * w_s - 1/2 * ||a||^2 * w_a
    where w_s and w_a are the weights for the state and action respectively.
    """
    if cost_func is None:
        cost_func = quad_cost
    n_total = ssa_samples.shape[1]
    n_state = (n_total - n_action)//2
    
    # init reward stack 
    assert w_s >= 0 and w_s <= 1, "w_s should be between 0 and 1"
    w_a = np.sqrt(1 - w_s**2)
    rew = cost_func(ssa_samples, w_s, n_action, n_state, w_a)
    # print(f"rew.shape", rew.shape)
    
    return np.concatenate([ssa_samples, rew], axis=1)


def create_ssa_samples(params, states, s_target, n_samples = 1000, normalize = True, n = 4):
    """
    shuffle the states and params, and create samples"""
    
    
    pair_idx = np.random.randint(0, len(states), size = (n_samples, 2))
    ssa_samples= []
    
    for i in range(n_samples): 
        # print(f"pair_idx: {pair_idx[i]}")
        state = states[pair_idx[i, 0]]
        next_state = states[pair_idx[i, 1]]
        n_group = min(n, len(state))
        within_param_set_idx = np.random.randint(0, len(state)-n_group + 1) # avg group mean 
        state= np.mean(state[within_param_set_idx: within_param_set_idx + n_group], axis = 0) # calculate mean of a consecutive n_group samples
        n_group = min(n, len(next_state))
        within_param_set_idx = np.random.randint(0, len(next_state)-n_group + 1)
        next_state= np.mean(next_state[within_param_set_idx: within_param_set_idx + n_group], axis = 0)

        action = params[pair_idx[i, 1]] - params[pair_idx[i, 0]]

        ssa_samples.append( np.concatenate([state - s_target, next_state - s_target, action], axis = 0))
    ssa_samples = np.array(ssa_samples)
    # print(f"ssa_samples.shape: {ssa_samples.shape}")
    if normalize:
        # Normalize the samples
        n_s = states[0].shape[1]
        mean_s, std_s = np.mean(ssa_samples[:, :n_s], axis=0), np.std(ssa_samples[:, :n_s], axis=0)
        mean_a, std_a = np.mean(ssa_samples[:, 2* n_s:], axis=0), np.std(ssa_samples[:, 2*n_s:], axis=0)
        # normalize states and action respectively
        ssa_samples[:, :n_s] = (ssa_samples[:, :n_s] - mean_s) / std_s
        ssa_samples[:, n_s:2*n_s] = (ssa_samples[:, n_s:2*n_s] - mean_s) / std_s
        ssa_samples[:, 2*n_s:] = (ssa_samples[:, 2*n_s:]) / std_a # for action, we only normalize by std, not mean, because action is centered around 0
        return ssa_samples, [mean_s, mean_a], [std_s, std_a]

    return ssa_samples, [], []


def save_parameters_and_states(folder_path, params, names, states):
    np.save(os.path.join(folder_path, "params.npy"), params)
    np.save(os.path.join(folder_path, "param_names.npy"), names)
    for i in range(len(states)):
        state = states[i]
        np.save(os.path.join(folder_path, f"state_{i}.npy"), states[i])

def moving_average(data_list, avg_gait_num):
    """
    Calculate the moving average of the data list.
    """
    
    if len(data_list) < avg_gait_num:
        return data_list
    else:
        # print(f"moving average of {len(data_list)} gait cycles with {avg_gait_num} gait cycles, with {[dat.shape for dat in data_list]} shape")
        # For each window, concatenate DataFrames horizontally, then group by column name and average.
        # This correctly averages columns with the same name from different DataFrames in the window.
        # Concatenate DataFrames horizontally, then group by column name and average, keeping only one column per name
        return [
            pd.concat([df.reset_index(drop=True) for df in data_list[i : i + avg_gait_num]], axis=1)
            .T.groupby(by=lambda x: x.split('.')[0]).mean().T
            for i in range(len(data_list) - avg_gait_num + 1)
        ]

def print_state_stats(params, states, names=None):
    # Prepare column names as string representations of params
    param_names = [", ".join([f"{p:.2f}" for p in param]) for param in params]

    # Prepare state names
    if names is None:
        n_states = states[0].shape[1]
        state_names = [f"State {j}" for j in range(n_states)]
    else:
        state_names = names

    # Compute mean and std for each state variable for each param set
    data = []
    for j in range(len(state_names)):
        row = []
        for i in range(len(states)):
            avg = states[i].mean(axis=0)
            std = states[i].std(axis=0)
            row.append(f"{avg[j]:.2f}±{std[j]:.2f}")
        data.append(row)

    df = pd.DataFrame(data, index=state_names, columns=param_names)
    print("State means±std for each param set (columns are param sets):")
    print(df.to_string())

def skip_outliers(data, sw_st_idx, cfg, variable_name, test = False):
    """
    Skip the outliers in the data
    by using the mean and std of the selected variable
    return the list of data for each gait cycle, and the mean and std of the selected variable
    """
    data_list = []
    
    for i in range(len(sw_st_idx)-1):
        # skip the first few gait cycles
        if i < cfg["num_skip_gait_cycles"]:
            continue
        # skip the gait cycles that shorter than 0.5s
        if (sw_st_idx[i+1] - sw_st_idx[i]) < cfg[ "min_timesteps_per_gait_cycle"]:
            print(f"skip gait cycle {i} because of too short")
            continue
        
        interpolated_data = data.iloc[sw_st_idx[i]:sw_st_idx[i+1]].copy()
        interpolated_data = interpolated_data.interpolate(method='linear', limit_direction='both')
        # interpolate the data to 100 points
        interpolated_data = interpolated_data.iloc[np.linspace(0, len(interpolated_data)-1, 100).astype(int)]
        # add hip velocity and shank velocity in the interpolated data
        interpolated_data["ACTUATOR_POSITION"] = interpolated_data["ACTUATOR_POSITION"].rolling(window=5, center=True, min_periods=1).mean()
        interpolated_data["SHANK_ANGLE"] = interpolated_data["SHANK_ANGLE"].rolling(window=5, center=True).mean()
        interpolated_data["HIP_ANGLE"] = interpolated_data["ACTUATOR_POSITION"] - interpolated_data["SHANK_ANGLE"]
        interpolated_data["HIP_VELOCITY"] = np.gradient(interpolated_data["HIP_ANGLE"].rolling(window=5, center=True, min_periods=1).mean(), 0.1) # Smooth data with rolling mean
        interpolated_data["SHANK_VELOCITY"] = np.gradient(interpolated_data["SHANK_ANGLE"].rolling(window=5, center=True, min_periods=1).mean(), 0.1)
        interpolated_data["KNEE_VELOCITY"] = np.gradient(interpolated_data["ACTUATOR_POSITION"].rolling(window=5, center=True, min_periods=1).mean(), 0.1)
        # # skip the gait cycles that stance and swing are inproportional
        if np.sum(interpolated_data["GAIT_PHASE"]) >  70 or np.sum(interpolated_data["GAIT_PHASE"]) < 30:
            print(f"skip gait cycle {i} because of inproportional phase")
            continue
        
        if test:
            print("Entering test mode")
            # plot the data by frame in sagittal plane
            # st_ratio = np.sum(interpolated_data["LOADCELL"] > cfg["loadcell_threshold"]) / len(interpolated_data)
            st_ratio = np.sum(1 - interpolated_data["GAIT_PHASE"]) / len(interpolated_data)
            plot_leg_sagittal_plane(interpolated_data["ACTUATOR_POSITION"] -20, interpolated_data["SHANK_ANGLE"], st_ratio=st_ratio, skip_rate=10)
            
        # np.save( "knee_angle.npy", interpolated_data["ACTUATOR_POSITION"])
        # np.save( "shank_angle.npy", interpolated_data["SHANK_ANGLE"])
        data_list.append(interpolated_data)
    
    if len(data_list) > 0: 
        data_KA = np.array(pd.concat(data_list, axis=0)[variable_name]).reshape(-1, 100)
        
        mean, std = np.mean(data_KA, axis=0), np.std(data_KA, axis=0)
        # skip the knee angle outside mean +- 3* std
        mask = np.all(np.abs(data_KA - mean) <= 3 * std, axis=1)
        
        data_list_skip = [data_list[i] for i in range(len(data_list)) if mask[i]]

        return data_list_skip
    else:
        print(f"no data")
        return []



def extract_gait_data(f,cfg, filter_kernel_size = 5):
    """
    Extract gait data from the given file.
    Filter the LOADCELL data using a moving average filter.
    Infer the stance or swing phase based on the filtered LOADCELL data.
    return the data with filtered LOADCELL data and the SW to ST index of the gait cycle (start of each gait cycle)
    """
    # print(f)
    raw_data = pd.read_csv(f, header=0, skiprows=0)
                # Convert numeric columns to float, ignoring non-numeric data
    raw_data = raw_data.apply(pd.to_numeric, errors='coerce')
                # Drop rows with any NaN values resulting from non-numeric data
    raw_data = raw_data.dropna()    
    raw_data = raw_data[(raw_data > -1e6).all(axis=1) & (raw_data < 1e6).all(axis=1)]
    
    filtered_loadcell = raw_data["LOADCELL"] #np.convolve(raw_data["LOADCELL"].values, np.ones(filter_kernel_size)/filter_kernel_size, mode='same')
    raw_data["LOADCELL"] = filtered_loadcell

    if "GAIT_PHASE" not in raw_data.columns:
        print(f"no gait phase for {f}, infering from subphase")
        raw_data["GAIT_PHASE"] = raw_data["GAIT_SUBPHASE"] > 2.
    # infer the stance or swing phase by loadcell data:
    # phase = np.where(filtered_loadcell >= cfg["loadcell_threshold"], 1, 0)
    phase = 1 - raw_data["GAIT_PHASE"].values
    
    # phase = np.where(raw_data["GAIT_SUBPHASE"].values >= 1, 0, 1) 
    # phase = np.where(raw_data["GAIT_SUBPHASE"].values <= 2, 1, 0) 
    sw_st_idx = np.where(phase[:-1] - phase[1:] == -1)[0] + 1
    st_sw_idx = np.where(phase[:-1] - phase[1:] == 1)[0] + 1

    phase_var = np.zeros(len(phase))
    for i in range(len(sw_st_idx)-1):
        phase_var[sw_st_idx[i]:sw_st_idx[i+1]] = np.linspace(0, 1, sw_st_idx[i+1] - sw_st_idx[i])
                
    # clip to keep only complete gait cycles
    data, phase_var, phase = raw_data.iloc[sw_st_idx[0]:sw_st_idx[-1]], phase_var[sw_st_idx[0]:sw_st_idx[-1]], phase[sw_st_idx[0]:sw_st_idx[-1]]
    sw_st_idx = sw_st_idx[1:] - sw_st_idx[0]
    st_sw_idx = st_sw_idx[st_sw_idx > sw_st_idx[0]] - sw_st_idx[0]
    return data,sw_st_idx, st_sw_idx


def load_from_data(ssar, n_state = 4, n_action = 3):
    samples = []
    assert n_state *2 + n_action + 1 == ssar.shape[1], "ssar should have shape (n_samples, n_state*2 + n_action + 1)"
    for i in range(len(ssar)):
        state = ssar[i][:n_state].copy()
        next_state = ssar[i][n_state:n_state*2].copy()
        action = ssar[i][n_state*2:-1].copy()
        reward = ssar[i][-1].copy()
        done = False
        samples.append(Sample(state, action, reward, next_state, done))
    return samples

def load_samples_from_file(filename):
    samples = []
    with open(filename, 'r') as file:
        for line in file:
            data = line.strip().split(',')
            state = np.array([float(x) for x in data[0:4]])
            action = int(data[4])
            reward = float(data[5])
            next_state = np.array([float(x) for x in data[6:10]])
            done = bool(int(data[10]))
            samples.append(Sample(state, action, reward, next_state, done))
    return samples

def generate_file(file_name, num_samples=1000):
    with open(file_name, 'w') as file:
        for _ in range(num_samples):
            state = np.random.rand(4)
            action = np.random.randint(0, 2)
            reward = np.random.rand()
            next_state = np.random.rand(4)
            done = np.random.choice([0, 1])
            file.write(f"{','.join(map(str, state))},{action},{reward},{','.join(map(str, next_state))},{done}\n")

def generate_samples(states, params, num_samples=1000):
    pairs_idx = np.random.choice(np.arange(len(params)), size=(3000, 2), replace=True)
    
    print(f"params: {params}, pairs_idx: {pairs_idx}")
    samples = []
    for _ in range(num_samples):
        state = np.random.rand(4)
        action = np.random.randint(0, 2)
        reward = np.random.rand()
        next_state = np.random.rand(4)
        done = np.random.choice([0, 1])
        samples.append(Sample(state, action, reward, next_state, done))
    return samples