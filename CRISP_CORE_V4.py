# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# CRISP: Cremated Remains Inference of Sex Probabilities - CORE
# Created and maintained by Lukas Waltenberger, PhD
# Protected under GPL-3.0
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import math

# Data entry:
# Parameters for Bayiesian model creation
parameter_table = pd.DataFrame({
    "metric_vars": [
        "Mandible: condyle width", "Axis: ant.-post. diameter", "Humerus: vert. head diameter",
        "Humerus: trochlea max. diameter", "Humerus: trochlea min. diameter", "Humerus: capitolum max. diameter",
        "Radius: max. head diameter", "Lunate: max. width", "Lunate: max. length", "Femur: vert. Head diameter",
        "Patella: max. height", "Patella: max. width", "Patella: max. thickness", "Talus: max. length",
         "Talus: trochlea length", "Talus: trochlea width", "Navicular: max. length",
        "MT1: dorsoplantar width of the head", "MT1: med.-lat. width of the head"
    ],
    "mean_m": [
        16.9515789473684, 9.96882352941176, 40.6314285714286, 20.349375, 13.7510714285714, 17.0741666666667,
        19.767037037037, 14.7261538461538, 14.2344444444444, 42.1445454545455, 38.98, 37.73125, 16.6148,
        48.8416666666667, 31.5153846153846, 29.7008333333333, 13.9966666666667, 17.2260714285714, 2.93347655214123
    ],
    "sd_m": [
        1.43575865798421, 0.938870258264598, 2.71468331385182, 2.33251071808899,  1.71907455569627, 1.32848141830734,
        1.28542065791953, 1.31627339144482, 1.12317753617929,  3.14684401785778, 2.17005888296967, 3.37617466330503,
        2.23701274024088, 2.41477466167481, 2.3684650985558, 2.90962557977702, 2.46177829646627, 1.41509936645142,
        0.09792085711275
    ],
    "mean_f": [
        14.7731578947368, 8.90307692307692, 35.5047368421053, 18.6675, 12.1355882352941, 15.595, 17.0684615384615,
        13.235, 12.0135294117647, 36.2629411764706, 34.6113333333333, 33.7566666666667, 14.66, 44.9318181818182,
        26.870625, 25.9755882352941, 11.9408695652174, 15.2817142857143, 2.78787854916411
    ],
    "sd_f": [
        1.57559883723822, 0.804043626830133, 2.1958530825704, 1.3606147024125,  1.28923324917041, 1.92545752138723,
        1.27692174771893, 1.27598446841502, 2.10441779708923, 3.43652397908461, 1.46221391178224, 2.35089851110672,
        1.73313616764805, 2.49997927264135, 2.17680949633479, 2.28709447960623, 1.6148175045876, 1.29200593177086,
        0.10128564242292
    ]
})

# Expected columns
expected_columns = [
    "Site", "Grave", "Mandible: condyle width", "Axis: ant.-post. diameter",  "Humerus: vert. head diameter",
    "Humerus: trochlea max. diameter", "Humerus: trochlea min. diameter", "Humerus: capitolum max. diameter",
    "Radius: max. head diameter", "Lunate: max. width", "Lunate: max. length", "Femur: vert. Head diameter",
    "Patella: max. height", "Patella: max. width", "Patella: max. thickness", "Talus: max. length",
    "Talus: trochlea length", "Talus: trochlea width", "Navicular: max. length",  "MT1: dorsoplantar width of the head",
    "MT1: med.-lat. width of the head", "Notes", "Probability female", "Probability male", "Warnings"
]

var_means = {
    "Mandible: condyle width": 15.862368, "Axis: ant.-post. diameter": 9.507000,
    "Humerus: vert. head diameter": 37.679697, "Humerus: trochlea max. diameter": 19.415000,
    "Humerus: trochlea min. diameter": 12.865161, "Humerus: capitolum max. diameter": 16.228929,
    "Radius: max. head diameter": 18.172424, "Lunate: max. width": 14.010400, "Lunate: max. length": 12.782308,
    "Femur: vert. Head diameter": 38.573571, "Patella: max. height": 36.358800, "Patella: max. width": 35.139130,
    "Patella: max. thickness": 15.389403, "Talus: max. length": 46.311765,  "Talus: trochlea length": 28.952759,
    "Talus: trochlea width": 27.517069, "Navicular: max. length": 12.752368,
    "MT1: dorsoplantar width of the head": 16.145873, "MT1: med.-lat. width of the head": 17.473673
}

var_sd = {
    "Mandible: condyle width": 1.8517551, "Axis: ant.-post. diameter": 1.0249866,
    "Humerus: vert. head diameter": 3.5109298, "Humerus: trochlea max. diameter": 2.0137194,
    "Humerus: trochlea min. diameter": 1.6923829, "Humerus: capitolum max. diameter": 1.8260184,
    "Radius: max. head diameter": 1.8443511, "Lunate: max. width": 1.4800802, "Lunate: max. length": 2.0973799,
    "Femur: vert. Head diameter": 4.3843790, "Patella: max. height": 2.7900632, "Patella: max. width": 3.3000329,
    "Patella: max. thickness": 2.1431384, "Talus: max. length": 3.0720906, "Talus: trochlea length": 3.2355678,
    "Talus: trochlea width": 3.1416109, "Navicular: max. length": 2.2092147,
    "MT1: dorsoplantar width of the head": 1.6540826, "MT1: med.-lat. width of the head": 2.1749231
}

var_upper_border = {}
var_lower_border = {}

for var in parameter_table['metric_vars']:
    mean_val = var_means[var]
    sd_val = var_sd[var]
    var_upper_border[var] = mean_val + 3 * sd_val
    var_lower_border[var] = mean_val - 3 * sd_val

# Bayesian model
prior_m = 0.5
prior_f = 0.5

# Functions for usage in GUI:

# Normal distribution - density function
# TODO: Make own norm.pdf func, remove scipy import
def norm_prob(x, mu, sd):
    try:
        return norm.pdf(x, loc=mu, scale=sd)
    except Exception as e:
        print(e)
        return None

#  Likelihood function
# TODO: remove call for parameter_table
def add_likelihood(var_name, value, parameter_table):
    row = parameter_table[parameter_table["metric_vars"] == var_name]

    if row.empty:
        raise ValueError(f"Error: Variable '{var_name}'not found in parameter_table")

    mu_m = row["mean_m"].values[0]
    sd_m = row["sd_m"].values[0]
    mu_f = row["mean_f"].values[0]
    sd_f = row["sd_f"].values[0]

    prob_m = norm_prob(value, mu=mu_m, sd=sd_m)
    prob_f = norm_prob(value, mu=mu_f, sd=sd_f)

    return {
        "log_m": np.log(prob_m),
        "log_f": np.log(prob_f),
        "prob_m": prob_m,
        "prob_f": prob_f
    }

# TODO: remove call for parameter_table
def consistency_check(individual_posteriors, parameter_table, new_case):
    male_probs = np.array(list(individual_posteriors.values()))
    used_vars = list(individual_posteriors.keys())
    n_used = len(used_vars)

    warning_str = ""

    if n_used >= 5:
        median_prob = np.median(male_probs)
        deviations = np.abs(male_probs - median_prob)

        if np.any(deviations > 0.25):
            warning_str = warning_str + (f"Warning: Inconsistency for ≥5 used features!\n   → Median "
                                         f"Posterior(m): {median_prob:.2f}\n   → deviation > 0.25 detected:\n   "
                                         f"→ all individual posterior(m) values (ascending):\n")

            sorted_indices = np.argsort(male_probs)
            sorted_vars = [used_vars[i] for i in sorted_indices]
            sorted_vals = male_probs[sorted_indices]

            for name, val in zip(sorted_vars, sorted_vals):
                mark = "*" if abs(val - median_prob) > 0.25 else ""
                warning_str = warning_str + f"      {name:<25} : {val:.3f} {mark}\n"

            warning_str = warning_str + "   * = deviation > 0.25\n"

    elif n_used >= 2:
        max_diff = np.max(male_probs) - np.min(male_probs)
        if max_diff > 0.33:
            warning_str = warning_str + f"Warning: Inconsistency for few features (Δ = {max_diff:.3f})\n   → single posterior (m):\n"
            for name, val in zip(used_vars, male_probs):
                warning_str = warning_str + f"      {name}: {val:.4f}\n"
    else:
        warning_str = warning_str + "Only one feature available; no inconsistency check.\n"

    # Check for measurement errors (outside 3 SD)
    for var in parameter_table['metric_vars']:
        val = new_case.iloc[0][var]
        if val is not None and not pd.isna(val):
            if val > var_upper_border[var] or val < var_lower_border[var]:
                warning_str = warning_str + f"Warning: {var} outside of 3 SDs\n"

    return warning_str

# Makes data frame from dict
# TODO: data legality check
def new_single_case(data = {}):
    new_case = pd.DataFrame([data])
    return new_case

# Read cases from an excel file
# TODO: data legality check
def new_raw_cases_file(file_path):
    if file_path:
        df = pd.read_excel(file_path)
        print(f"data loaded: {file_path}")
        #print(df.head())
        return df
    else:
        print("no files chosen")
        return None

# Prepares the raw data frame for calculations
def file_cases(file_path):
    cases_raw = new_raw_cases_file(file_path)
    warn_str = ""
    critical = False
    if cases_raw is not None:
        missing_columns = [col for col in expected_columns if col not in cases_raw.columns]
        if missing_columns:
            warn_str = warn_str + f"Error: Missing variables: {missing_columns}\n"
            critical = True
        else:
            excel_part2 = cases_raw[parameter_table["metric_vars"]].copy()
            excel_part2 = excel_part2.apply(pd.to_numeric, errors='coerce')  # change all to numeric
            mask_invalid = (excel_part2 <= 0)  # check neg values and 0 in multiple data entry
            if mask_invalid.any().any():
                warn_str = warn_str + "Warning: Negative value or 0 detected!"
                excel_part2 = excel_part2.mask(mask_invalid, np.nan)

            return excel_part2, cases_raw, warn_str, critical
    return None, None, warn_str, critical

# Prediction of single cases
# Return probabilities and graphs for single cases
def calculate_single(raw_data):
    single_case = new_single_case(raw_data)
    var_log = "MT1: med.-lat. width of the head"
    if var_log in single_case.columns:
        val = single_case.at[0, var_log]
        if val is not None and not pd.isna(val) and val > 0:
            single_case.at[0, var_log] = np.log(val)
        else:
            single_case.at[0, var_log] = np.nan

    log_likelihood_m = np.log(prior_m)
    log_likelihood_f = np.log(prior_f)
    individual_posteriors = {}
    # Evaluation of all variables
    for var in parameter_table['metric_vars']:
        value = single_case.iloc[0][var]  # new case to predict
        if value is not None and not pd.isna(value):
            contrib = add_likelihood(var, value, parameter_table)

            log_likelihood_m += contrib["log_m"]
            log_likelihood_f += contrib["log_f"]

            posterior_m = (contrib["prob_m"] * prior_m) / (
                    (contrib["prob_m"] * prior_m) + (contrib["prob_f"] * prior_f)
            )
            individual_posteriors[f"{var}.m"] = posterior_m

    likelihoods = np.exp([log_likelihood_m, log_likelihood_f])
    posterior = likelihoods / np.sum(likelihoods)

    # consistency check
    warn_str = consistency_check(individual_posteriors, parameter_table, single_case)

    # results
    #print("\n>>> probability (total):")
    #print(f"male: {posterior[0] * 100:.2f}%")
    #print(f"female: {posterior[1] * 100:.2f}%")

    plot_names = [
        "Mandible: condyle width", "Axis: A-P diameter", "Humerus: vert. head d.", "Humerus: trochlea max. d.",
        "Humerus: trochlea min. d.", "Humerus: capitolum max. d.", "Radius: max. head d.", "Lunate: max. width",
        "Lunate: max. length", "Femur: vert. Head d.", "Patella: max. height", "Patella: max. width",
        "Patella: max. thickness", "Talus: max. length", "Talus: trochlea length", "Talus: trochlea width",
        "Navicular: max. length", "MT1: head dorsoplantar w.", "MT1: head: med.-lat. w. (log)"
    ]

    # density plots
    n_cols = 5  # number of columns in plot grid

    plt.rcParams.update({
        'font.size': 4,
        'axes.titlesize': 8,
        'axes.labelsize': 7,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'legend.fontsize': 7
    })

    plot_vars = [var for var in parameter_table['metric_vars'] if var in single_case.columns]
    n_plots = len(plot_vars)
    n_rows = math.ceil(n_plots / n_cols)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
    axs = axs.flatten()

    for i, var in enumerate(plot_vars):
        stats = parameter_table[parameter_table['metric_vars'] == var]
        if stats.shape[0] == 1:
            stats = stats.iloc[0]

            x_min = min(stats['mean_m'] - 4 * stats['sd_m'], stats['mean_f'] - 4 * stats['sd_f'])
            x_max = max(stats['mean_m'] + 4 * stats['sd_m'], stats['mean_f'] + 4 * stats['sd_f'])
            x_vals = np.linspace(x_min, x_max, 500)

            female_density = norm.pdf(x_vals, loc=stats['mean_f'], scale=stats['sd_f'])
            male_density = norm.pdf(x_vals, loc=stats['mean_m'], scale=stats['sd_m'])

            ax = axs[i]
            ax.plot(x_vals, female_density, color='red', label='Female')
            ax.plot(x_vals, male_density, color='blue', label='Male')
            ax.set_title(plot_names[i])
            # ax.set_xlabel("value [mm]")
            # ax.set_ylabel("density")

            val = single_case.at[0, var] if var in single_case.columns else None
            if pd.notna(val) and val > 0:
                ax.axvline(val, color='black', linestyle='dashed', linewidth=2)

    # legend only in last subplot
    handles, labels = axs[0].get_legend_handles_labels()  # take label from one plot

    legend_ax = axs[-1]
    legend_ax.axis('off')  # no axis in legend
    legend_ax.legend(handles, labels, loc='center', fontsize=8)  # center legend

    # delete further empty subplots
    for j in range(i + 1, len(axs) - 1):
        axs[j].axis('off')
        plt.tight_layout(pad=3.0)

    fig.subplots_adjust(wspace=0.3, hspace=0.5)  # adjust space between plots

    # adjust size of plot
    fig.set_size_inches(7, 5)  # 700x500 px with dpi=100

    return posterior, warn_str, fig





    # density plots
    # n_cols = 5  # number of columns in plot grid
    #
    # plt.rcParams.update({
    #     'font.size': 4,
    #     'axes.titlesize': 9,
    #     'axes.labelsize': 8,
    #     'xtick.labelsize': 7,
    #     'ytick.labelsize': 7,
    #     'legend.fontsize': 7
    # })
    #
    # plot_vars = [var for var in parameter_table['metric_vars'] if var in single_case.columns]
    # n_plots = len(plot_vars)
    # n_rows = math.ceil(n_plots / n_cols)
    #
    # fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
    # axs = axs.flatten()
    #
    # for i, var in enumerate(plot_vars):
    #     stats = parameter_table[parameter_table['metric_vars'] == var]
    #     if stats.shape[0] == 1:
    #         stats = stats.iloc[0]
    #
    #         x_min = min(stats['mean_m'] - 4 * stats['sd_m'], stats['mean_f'] - 4 * stats['sd_f'])
    #         x_max = max(stats['mean_m'] + 4 * stats['sd_m'], stats['mean_f'] + 4 * stats['sd_f'])
    #         x_vals = np.linspace(x_min, x_max, 500)
    #
    #         female_density = norm.pdf(x_vals, loc=stats['mean_f'], scale=stats['sd_f'])
    #         male_density = norm.pdf(x_vals, loc=stats['mean_m'], scale=stats['sd_m'])
    #
    #         ax = axs[i]
    #         ax.plot(x_vals, female_density, color='red', label='Female')
    #         ax.plot(x_vals, male_density, color='blue', label='Male')
    #         ax.set_title(var)
    #         ax.set_xlabel("value [mm]")
    #         ax.set_ylabel("density")
    #
    #         val = single_case.at[0, var] if var in single_case.columns else None
    #         if pd.notna(val) and val > 0:
    #             ax.axvline(val, color='black', linestyle='dashed', linewidth=2)
    #
    #         ax.legend()
    #
    # # Deactivate unused subplots
    # for j in range(i + 1, len(axs)):
    #     axs[j].axis('off')
    #     fig.set_size_inches(700/100, 500/100)
    #
    #
    # return posterior, warn_str, fig

# Multiple data entry
# Return dataframe
def calculate_file(excel_part2, cases_raw):
    # Log-Transformation
    excel_part2["MT1: med.-lat. Width of the head"] = np.log(excel_part2["MT1: med.-lat. Width of the head"])

    posterior_all = []
    warnings = []

    for idx, row in excel_part2.iterrows():
        new_case = row
        log_likelihood_m = np.log(prior_m)
        log_likelihood_f = np.log(prior_f)
        male_probs = {}

        for var in parameter_table["metric_vars"]:
            value = new_case[var]
            if pd.notna(value):
                contrib = add_likelihood(var, value, parameter_table)
                log_likelihood_m += contrib['log_m']
                log_likelihood_f += contrib['log_f']
                male_probs[var] = contrib['prob_m'] / (contrib['prob_m'] + contrib['prob_f'])

        likelihoods = np.exp([log_likelihood_m, log_likelihood_f])
        posterior = likelihoods / np.sum(likelihoods)
        posterior_dict = {'m': posterior[0], 'f': posterior[1]}

        # warnings
        warn_text = None
        n_used = len(male_probs)

        if n_used >= 5:
            median_prob = np.median(list(male_probs.values()))
            deviations = np.abs(np.array(list(male_probs.values())) - median_prob)
            if np.any(deviations > 0.25):
                sorted_vars = sorted(male_probs, key=male_probs.get)
                sorted_vals = [male_probs[v] for v in sorted_vars]
                warn_lines = [
                    "Inconsistency for ≥5 used features!",
                    f"   → Median Posterior(m): {median_prob:.3f}",
                    "   → deviation > 0.25 detected:",
                    "   → all individual posterior(m) values (ascending):"
                ]
                for name, val in zip(sorted_vars, sorted_vals):
                    mark = "*" if abs(val - median_prob) > 0.25 else ""
                    warn_lines.append(f"      {name:<25} : {val:.3f} {mark}")
                warn_lines.append("   * = deviation > 0.25")
                warn_text = "\n".join(warn_lines)
        elif n_used >= 2:
            max_diff = max(male_probs.values()) - min(male_probs.values())
            if max_diff > 0.33:
                sorted_probs = sorted(male_probs.items(), key=lambda x: x[1])
                warn_vals = "; ".join(f"{k}: {v:.3f}" for k, v in sorted_probs)
                warn_text = (f"Inconsistency for few features | Δ = {max_diff:.3f} | "
                            f"Posterior(m): {warn_vals}")

        # Check for measurement errors (outside 3 SD)
        for var_model in parameter_table["metric_vars"]:
            raw_val = cases_raw.at[idx, var_model] #compare data input and not log_trans
            try:
                val = float(str(raw_val).replace(",", "."))  #replace , through. if neccessary
                upper = var_upper_border.get(var_model, np.inf)
                lower = var_lower_border.get(var_model, -np.inf)
                if pd.notna(val)and (val > upper or val < lower):
                        msg = f"Measurement error: {var_model} = {val:.2f} outside 3 SD range ({lower:.2f} - {upper:.2f})"
                        if warn_text:
                            warn_text += " | " + msg
                        else:
                            warn_text = msg
            except Exception as e:
                print(f"Could not process {var_model} with value '{raw_val}': {e}")

        posterior_all.append({**posterior_dict, "warning": warn_text})
        if warn_text is not None:
            warnings.append(warn_text)

    # Results
    prob_male = [res['m'] for res in posterior_all]
    prob_female = [res['f'] for res in posterior_all]
    warnings_list = [res['warning'] if res['warning'] else "" for res in posterior_all]

    # update excel
    cases_raw["Probability male"] = [res['m'] for res in posterior_all]
    cases_raw["Probability female"] = [res['f'] for res in posterior_all]
    cases_raw["Warnings"] = [res['warning'] if res['warning'] else "" for res in posterior_all]

    return cases_raw
