import pandas as pd
import numpy as np

# Define the search_list
search_list = {
    "us": ["us", "hml", 2],
    "developed": ["developed", "hml", 2],
    "emerging": ["emerging", "hml", 2],
    "all": [["us", "developed", "emerging"], "hml", 3],
    "world": ["world", "hml", 2],
    "world_ex_us": ["world_ex_us", "hml", 2],
    "us_mega": ["us", "cmp", 2, "mega"],
    "us_large": ["us", "cmp", 2, "large"],
    "us_small": ["us", "cmp", 2, "small"],
    "us_micro": ["us", "cmp", 2, "micro"],
    "us_nano": ["us", "cmp", 2, "nano"]
}

# Define the empirical Bayes estimation
eb_est = {}
for key, x in search_list.items():
    print(f"Region: {x[0]}")
    regions = x[0]
    
    # Select the appropriate data
    if x[1] == "cmp":
        base_data = regional_pfs_cmp[regional_pfs_cmp['size_grp'] == x[3]].copy()
    elif x[1] == "hml":
        base_data = regional_pfs.copy()
    
    # Prepare data
    data = base_data[(base_data['eom'] >= settings['start_date']) & (base_data['eom'] <= settings['end_date']) & (base_data['region'].isin(regions))]
    data = eb_prepare(data, scale_alphas=settings['eb']['scale_alpha'], overlapping=settings['eb']['overlapping'])
    
    # Run empirical Bayes estimation
    op = emp_bayes(
        data, 
        cluster_labels=cluster_labels,
        min_obs=settings['eb']['min_obs'],
        fix_alpha=settings['eb']['fix_alpha'],
        bs_cov=settings['eb']['bs_cov'],
        layers=x[2],
        shrinkage=settings['eb']['shrinkage'],
        cor_type=settings['eb']['cor_type'],
        bs_samples=settings['eb']['bs_samples'],
        seed=settings['seed']
    )
    
    # Store the result
    eb_est[key] = op

# Simulations EB vs. BY
if update_sim:
    # Pairwise correlation calculation
    pairwise_cor = eb_est['us']['input']['long'].pivot_table(index='eom', columns='characteristic', values='ret_neu').corr(method='pearson')
    
    cor_value = pairwise_cor.stack().reset_index(name='cor')
    cor_value = cor_value.merge(cluster_labels[['characteristic', 'hcl_label']].rename(columns={'hcl_label': 'hcl1'}), left_on='level_0', right_on='characteristic', how='left')
    cor_value = cor_value.merge(cluster_labels[['characteristic', 'hcl_label']].rename(columns={'hcl_label': 'hcl2'}), left_on='level_1', right_on='characteristic', how='left')
    cor_value['same_cluster'] = (cor_value['hcl1'] == cor_value['hcl2'])
    cor_value_avg = cor_value.groupby('same_cluster')['cor'].mean().reset_index(name='avg_cor')

    # Time periods
    med_months = eb_est['us']['input']['long'].groupby('characteristic')['eom'].count().median()
    data_sim = {
        'yrs': round(med_months / 12),
        'cor_within': round(cor_value_avg[cor_value_avg['same_cluster'] == True]['avg_cor'].values[0], 2),
        'cor_across': round(cor_value_avg[cor_value_avg['same_cluster'] == False]['avg_cor'].values[0], 2)
    }
    
    # Simulation Settings
    np.random.seed(settings['seed'])
    sim = {
        "alpha_0": 0,
        "t": 12 * 70,
        "clusters": 13,
        "fct_pr_cl": 10,
        "corr_within": 0.58,
        "corr_across": 0.02,
        "n_sims": 10000,
        "tau_c": [0.01] + list(np.arange(0.05, 0.55, 0.05)),
        "tau_w": [0.01, 0.2],
    }
    sim['se'] = (10 / np.sqrt(12)) / np.sqrt(sim['t'])
    sim['n'] = sim['clusters'] * sim['fct_pr_cl']

    # Check if simulation settings match data
    if (abs(sim['t'] - data_sim['yrs'] * 12) > 12 or
            abs(sim['corr_within'] - data_sim['cor_within']) > 0.05 or
            abs(sim['corr_across'] - data_sim['cor_across']) > 0.05):
        print("SIMULATION AND DATA VALUES ARE NOT CONSISTENT!")
        print(data_sim)
        print({"yrs": sim['t'] / 12, "corr_within": sim['corr_within'], "corr_across": sim['corr_across']})
    
    simulation = sim_mt_control(sim_settings=sim)
    simulation.to_pickle(f"{object_path}/fdr_sim.pkl")
else:
    simulation = pd.read_pickle(f"{object_path}/fdr_sim.pkl")

# False Discovery Rate
model_fdr = fdr_sim(t_low=0, a_vec=eb_est['us']['factor_mean'], a_cov=eb_est['us']['factor_cov'], n_sim=10000, seed=settings['seed'])

# Multiple Testing Adjustments
mt = multiple_testing(eb_all=eb_est['all'], eb_world=eb_est['world'])

# Tangency Portfolios
tpf_world = tpf_cluster(eb_est['world']['input']['long'], mkt_region='world', orig_sig=True, min_date=settings['tpf']['start']['world'],
                        n_boots=settings['tpf']['bs_samples'], shorting=settings['tpf']['shorting'], seed=settings['seed'])

tpf_us = tpf_cluster(eb_est['us']['input']['long'], mkt_region='us', orig_sig=True, min_date=settings['tpf']['start']['us'],
                     n_boots=settings['tpf']['bs_samples'], shorting=settings['tpf']['shorting'], seed=settings['seed'])

tpf_dev = tpf_cluster(eb_est['developed']['input']['long'], mkt_region='developed', orig_sig=True, min_date=settings['tpf']['start']['developed'],
                      n_boots=settings['tpf']['bs_samples'], shorting=settings['tpf']['shorting'], seed=settings['seed'])

tpf_emer = tpf_cluster(eb_est['emerging']['input']['long'], mkt_region='emerging', orig_sig=True, min_date=settings['tpf']['start']['emerging'],
                       n_boots=settings['tpf']['bs_samples'], shorting=settings['tpf']['shorting'], seed=settings['seed'])

# Size Groups
tpf_size = pd.concat([
    tpf_cluster(eb_est[f'us_{x}']['input']['long'], mkt_region='us', orig_sig=True, min_date=settings['tpf']['start']['size_grps'],
                n_boots=settings['tpf']['bs_samples'], shorting=settings['tpf']['shorting'], seed=settings['seed']).assign(size_grp=x)
    for x in ["mega", "large", "small", "micro", "nano"]
])

# Posterior Over Time
if update_post_over_time:
    for fix_taus in [True, False]:
        fixed_priors = None
        if fix_taus:
            fixed_priors = {
                "alpha": eb_est[ot_region]['mle'][eb_est[ot_region]['mle']['estimate'] == "alpha"]['ml_est'],
                "tau_c": eb_est[ot_region]['mle'][eb_est[ot_region]['mle']['estimate'] == "tau_c"]['ml_est'],
                "tau_s": eb_est[ot_region]['mle'][eb_est[ot_region]['mle']['estimate'] == "tau_s"]['ml_est']
            }

        periods = sorted(regional_pfs['eom'].unique())
        periods = [p for p in periods if p.month == 12]  # Only estimate once per year
        
        time_chars = regional_pfs[(regional_pfs['region'] == ot_region) & (regional_pfs['eom'] <= pd.Timestamp("1960-12-31"))]\
            .groupby('characteristic')['eom'].count().ge(settings['eb']['min_obs']).index.unique()
        
        posterior_over_time = []
        for end_date in periods[periods.index(pd.Timestamp("1960-12-31")):]:
            print(end_date)
            # Prepare Data
            data = regional_pfs[(regional_pfs['characteristic'].isin(time_chars)) &
                                (regional_pfs['eom'] >= settings['start_date']) & (regional_pfs['eom'] <= end_date) &
                                (regional_pfs['region'] == ot_region)]
            data = eb_prepare(data, scale_alphas=settings['eb']['scale_alpha'], overlapping=settings['eb']['overlapping'])
            
            # Run Empirical Bayes
            eb_act = emp_bayes(data, cluster_labels=cluster_labels, min_obs=settings['eb']['min_obs'], fix_alpha=settings['eb']['fix_alpha'],
                               bs_cov=settings['eb']['bs_cov'], layers=2, shrinkage=settings['eb']['shrinkage'], cor_type=settings['eb']['cor_type'],
                               bs_samples=1000, priors=fixed_priors, seed=settings['seed'])
            eb_act['input'] = None
            eb_act['end_date'] = end_date
            posterior_over_time.append(eb_act)

        # Save results
        filename = "posterior_over_time_fixed.pkl" if fix_taus else "posterior_over_time_flex.pkl"
        pd.to_pickle(posterior_over_time, f"{object_path}/{filename}")
else:
    posterior_over_time = pd.read_pickle(f"{object_path}/posterior_over_time_fixed.pkl")
    posterior_over_time_flex = pd.read_pickle(f"{object_path}/posterior_over_time_flex.pkl")

# Size Dimension
eb_us_size = pd.concat([
    eb_est[f'us_{x}']['factors'].assign(size_grp=x.capitalize())
    for x in ["mega", "large", "small", "micro", "nano"]
]).assign(size_grp=lambda df: pd.Categorical(df['size_grp'], categories=["Mega", "Large", "Small", "Micro", "Nano"]))

# In-Sample / Out-of-Sample
is_oos = {}
for t in ["pre", "post", "pre_post"]:
    data = prepare_is_oos(eb_est['us']['input']['long'], min_obs=60, ret_scaled="all", orig_group=True, type_=t, print_=True)
    regs = data.groupby(['characteristic', 'period', 'n_is', 'n_oos']).apply(lambda x: lm(x['ret_adj'], x['mkt_vw_exc']))
    regs['tidied'] = regs['fit'].apply(tidy)
    regs = regs[regs['tidied']['term'] == '(Intercept)']
    regs = regs.pivot_table(index=['characteristic', 'n_is', 'n_oos'], columns='period', values='estimate').reset_index()
    is_oos[t] = {'data': data, 'regs': regs}

# Save Posterior is Data
if update_post_is:
    posterior_is = []
    periods = sorted([p for p in regional_pfs['eom'].unique() if p.month == 12 and p.year >= 1959])

    for end_date in periods:
        print(f"Date {end_date} - {periods.index(end_date) + 1} out of {len(periods)}")
        # Prepare Data
        data = regional_pfs[(regional_pfs['eom'] >= settings['start_date']) & (regional_pfs['eom'] <= end_date) & (regional_pfs['region'] == 'us')]
        data = eb_prepare(data, scale_alphas=settings['eb']['scale_alpha'], overlapping=settings['eb']['overlapping'])

        # Run Empirical Bayes
        eb_act = emp_bayes(data, cluster_labels=cluster_labels, min_obs=settings['eb']['min_obs'], fix_alpha=settings['eb']['fix_alpha'],
                           bs_cov=settings['eb']['bs_cov'], layers=2, shrinkage=settings['eb']['shrinkage'], cor_type=settings['eb']['cor_type'],
                           bs_samples=1000, seed=settings['seed'])
        posterior_is.append(eb_act['factors'].assign(est_date=end_date))

    pd.to_pickle(pd.concat(posterior_is), f"{object_path}/posterior_is.pkl")
else:
    posterior_is = pd.read_pickle(f"{object_path}/posterior_is.pkl")

sig_oos_pfs = trading_on_significance(posterior_is)