############## SHOCK ADAPTATION FOOD SUPPLY MODEL #############
#
#
#  ADAPTED BY CSM_SEXY_GRP_ - 2025, ORIGIN: SOPHIA BAUM - 2024


### IMPORTS ### PAK RUS HOA URU ALL

from shock_input_data import *
import os

### PARAMETERS ###

scenario = 'ALL'                    # specify scenario
production_cap = True               # turn on / off global production cap
compensation = True                 # turn adaptation on
tau = 10                            # number of iterations
overshoot_data = []


input_folder  = './input/'          # folder with parameters and input data
output_folder = './results/'       # folder to write results to

overshoot_output_file = os.path.join(output_folder, 'production_overshoot.csv' if production_cap 
else 'total_prod.csv')
if os.path.exists(overshoot_output_file):
    all_overshoot_data = pd.read_csv(overshoot_output_file)
else:
    all_overshoot_data = pd.DataFrame()

limit_abs_sim = 1000                # Event limits
limit_rel_sim = 0.26
limit_dev_sim = 0.32


### LOADING DATA ###

shock_sectors, shock_scaling = create_shock_scaling_marix(scenario, tau)
                        # Load further information
io_codes = pd.read_csv(input_folder + 'io_codes_alph.csv').drop('Unnamed: 0', axis = 1)
su_codes = pd.read_csv(input_folder + 'su_codes_alph.csv').drop('Unnamed: 0', axis = 1)

                        # Create single indexes
areas     = np.array(sorted(set(io_codes['area'])))
items     = np.array(sorted(set(io_codes['item'])))
processes = np.array(sorted(set(su_codes['proc'])))

                        # Create multi index ('country', 'product')
ai_index = pd.MultiIndex.from_product([areas, items])

                        # Load  further information on areas (countries)
a_frame = pd.read_csv(input_folder + 'a_frame.csv')

                        # Counting
Ni = len(items)             # number of items
Np = len(processes)         # number of processes
Na = len(areas)             # number of countries

                        # Create a vector of all ones for summation
one_vec_proc = sprs.csr_matrix(np.ones(Na * Np))
one_vec_proc = one_vec_proc.transpose()

one_vec_item = sprs.csr_matrix(np.ones(Na * Ni))
one_vec_item = one_vec_item.transpose()

                        # Load initial conditions
vector_x0         = io.mmread(input_folder + '/sparse_x0.mtx')
vector_startstock = io.mmread(input_folder + '/sparse_startstock.mtx')

                        # Load model parameters
vector_eta_prod = io.mmread(input_folder + '/sparse_eta_prod.mtx')
vector_eta_exp  = io.mmread(input_folder + '/sparse_eta_exp.mtx')
matrix_nu       = io.mmread(input_folder + '/sparse_nu.mtx')
matrix_alpha    = io.mmread(input_folder + '/sparse_alpha.mtx')
matrix_beta     = io.mmread(input_folder + '/sparse_beta.mtx')
matrix_trade    = io.mmread(input_folder + '/sparse_trade.mtx')

                        # Turn data into sparse csr-format
x0          = sprs.csr_matrix(vector_x0)                       # initial condition
xstartstock = sprs.csr_matrix(vector_startstock)               # starting stock
eta_prod    = sprs.csr_matrix(vector_eta_prod)                 # allocation to production
eta_exp     = sprs.csr_matrix(vector_eta_exp)                  # allocation to trade
eta_cons    = one_vec_item - vector_eta_prod - vector_eta_exp  # allocation to neither production or trade (lost in model, summarized by consumption)
alpha       = sprs.csr_matrix(matrix_alpha)                    # conversion from input to output
beta        = sprs.csr_matrix(matrix_beta)                     # output for non-converting processes
T           = sprs.csr_matrix(matrix_trade)                    # fraction sent to each trading partner
nu          = sprs.csr_matrix(matrix_nu)                       # fraction allocated to a specific production process

                        # Eliminate zeros from sparse matrices
x0.eliminate_zeros()
xstartstock.eliminate_zeros()
eta_prod.eliminate_zeros()
eta_exp.eliminate_zeros()
eta_cons.eliminate_zeros()
alpha.eliminate_zeros()
beta.eliminate_zeros()
T.eliminate_zeros()
nu.eliminate_zeros()

                        # Determine countries that are producers of specific items
producer = (alpha@one_vec_proc) + (beta@one_vec_proc)
producer = producer.toarray()
producer = producer > 0
producer = pd.DataFrame(producer, index = ai_index, columns = ['is_producer'])
#producer.to_csv(output_folder + 'producer.csv')
#quit()

                        #Load adaptation rules

                        # Load transitions
for v in ['import','export','alpha','beta','nu','eta_exp','eta_prod','eta_cons']:
                            # Load transitions
        globals()['transition_' + v + '_multi'] = sprs.load_npz(input_folder+'transition_' + v + '_multi.npz')
        globals()['transition_' + v + '_multi'].data[globals()['transition_' + v + '_multi'].data < 0] = 0

        globals()['transition_' + v + '_rewire'] = sprs.load_npz(input_folder + 'transition_' + v + '_rewire.npz')
        globals()['transition_' + v + '_rewire'].data[globals()['transition_' + v + '_rewire'].data < 0] = 0
        
                            # Kick-Out all zero entries
        globals()['transition_' + v + '_multi' ].eliminate_zeros()
        globals()['transition_' + v + '_rewire'].eliminate_zeros()

                        #  Load subsitutability index
substitutability_trade = sprs.load_npz(input_folder + 'substitutability_trade.npz')
substitutability_trade.eliminate_zeros()


### SIMULATION: BASELINE ###

                        # Prepare storage
x         = sprs.csr_matrix((Na * Ni, 1))
x_timetrace_base = np.zeros((Na * Ni, tau))

                        # Set initial conditions
x = x0                          

                        # Iterate dynamics
for t in range(tau):          

    x = (alpha @ (nu @ (eta_prod.multiply(x))) + (beta @ one_vec_proc))  + T @ (eta_exp.multiply(x))

    if t == 0:
        x = x+xstartstock

                        # Update time series
    x_timetrace_base[:,t] = x.toarray()[:, 0]

                        # Store
xbase            = x.toarray()[:, 0]
X                = pd.DataFrame(xbase, index = ai_index, columns = ['base'])
X.index.names    = ['area', 'item']
X.columns.names  = ['scenario']

                        # Save
X.to_csv(output_folder + scenario + '_base_.csv')

                        # Output
print('Baseline scenario done.')


### SIMULATION: SHOCK ADAPTATION ###

                        # Prepare storage
XS                = pd.DataFrame(index = ai_index, columns = ['amount [t]'])
XS.index.names    = ['area','item']
XS.columns.names  = ['scenario']

                        # Find the shocked countries and items in the index
sector_ids = [list(ai_index.values).index(sector) for sector in shock_sectors]

                        # Initialize
xs_timetrace = np.zeros((Na * Ni, tau))
rl_timetrace = np.zeros((Na * Ni, tau))
al_timetrace = np.zeros((Na * Ni, tau))

xs = x0

alpha_shock    = alpha.copy()
beta_shock     = beta.copy()
nu_shock       = nu.copy()
eta_exp_shock  = eta_exp.copy()
eta_prod_shock = eta_prod.copy()
eta_cons_shock = eta_cons.copy()
T_shock        = T.copy()

xs_timetrace = np.zeros((Na * Ni, tau), dtype=np.float32)  # optional: comment out if not needed
rl_timetrace = np.zeros((Na * Ni, tau), dtype=np.float32)
al_timetrace = np.zeros((Na * Ni, tau), dtype=np.float32)


for t in range(tau):

    # Production
    o = (alpha_shock @ (nu_shock @ (eta_prod_shock.multiply(xs))) + (beta_shock @ one_vec_proc))

    # Start Stock
    if t == 0:
        o = o + xstartstock

    # Apply shocks
    for sector_id in sector_ids:
        sector = sector_ids.index(sector_id)
        o[sector_id] = shock_scaling[sector, t] * o[sector_id]

    if production_cap:
        initial_cap = 16e9
        productioncap = initial_cap * (1.011 ** t)
        initial_production = o.sum()
        print(f"Initial production was at {initial_production:.2f}")
        
        if initial_production > productioncap:
            scaling = productioncap / initial_production if initial_production > 0 else 0.0
            o = o.multiply(scaling)
            capped_prod = o.sum()
            print(f"Time {t}: Production capped at {productioncap:.2f}")
        else:
            scaling = 1.0
            print(f"No capping needed (scaling: {scaling:.4f})")
            

        overshoot_data.append({
            'scenario': scenario,
            'time_step': t,
            'initial production': float(initial_production),
            'cap': float(productioncap),
            'capped production' : float(capped_prod),
            'scaling': float(scaling)
            })

    else:
        # Track production without cap for this time step
        overshoot_data.append({
            'scenario': scenario,
            'time_step': t,
            'total_prod': float(xs.sum()),
            'cap': float('inf'),
            'scaling': 1.0
        })
    
    h = T_shock @ (eta_exp_shock.multiply(xs))
    print(f"Time {t}:Production (o): {o.sum():.2f}, Trade volume (h): {h.sum():.2f}")
    xs = o + h
    print(f"Time {t}: Total production (xs): {xs.sum():.2f}")
    xs_timetrace[:, t] = xs.toarray()[:, 0]

    # Relative loss
    rl = sprs.csr_matrix(np.nan_to_num(1 - xs / sprs.csr_matrix(x_timetrace_base[:, t]).T, nan=0))
    rl.data[rl.data < -1] = -1
    rl_timetrace[:, t] = rl.toarray()[:, 0]

    # Absolute loss
    al = sprs.csr_matrix(np.nan_to_num(sprs.csr_matrix(x_timetrace_base[:, t]).T - xs, nan=0))
    al_timetrace[:, t] = al.toarray()[:, 0]

    # Check for events
    if t == 1 and compensation:

        change_rl = set(np.where(rl.toarray()[:, 0] > limit_rel_sim)[0])  # rl_shock
        change_al = set(np.where(al.toarray()[:, 0] > limit_abs_sim)[0])  # al_shock
        change = np.array(list(change_rl.intersection(change_al)))
        mask = np.isin(np.arange(Na * Ni), change)

        alpha_shock[mask, :] = (alpha[mask, :].multiply(transition_alpha_multi[mask, :]) +
                                transition_alpha_rewire[mask, :]).multiply(rl[mask])
        mask_2 = (alpha_shock.sum(axis=0).A1 > 0) & (
            (alpha_shock.sum(axis=0).A1 < alpha.sum(axis=0).A1 * 0.99) |
            (alpha_shock.sum(axis=0).A1 > alpha.sum(axis=0).A1 * 1.01)
        )
        alpha_shock[:, mask_2] = alpha_shock[:, mask_2].multiply(
            alpha.sum(axis=0).A1[mask_2] / alpha_shock.sum(axis=0).A1[mask_2])

        beta_shock[mask, :] = (beta[mask, :].multiply(transition_beta_multi[mask, :]) +
                               transition_beta_rewire[mask, :]).multiply(rl[mask])
        mask_3 = (beta_shock.sum(axis=0).A1 > 0) & (
            (beta_shock.sum(axis=0).A1 < beta.sum(axis=0).A1 * 0.99) |
            (beta_shock.sum(axis=0).A1 > beta.sum(axis=0).A1 * 1.01)
        )
        beta_shock[:, mask_3] = beta_shock[:, mask_3].multiply(
            beta.sum(axis=0).A1[mask_3] / beta_shock.sum(axis=0).A1[mask_3])

        nu_shock[:, mask] = (nu[:, mask].multiply(transition_nu_multi[:, mask]) +
                             transition_nu_rewire[:, mask]).multiply(rl[mask].T)
        mask_4 = (nu_shock.sum(axis=0).A1 > 0) & (
            (nu_shock.sum(axis=0).A1 < 0.99) | (nu_shock.sum(axis=0).A1 > 1.01)
        )
        nu_shock[:, mask_4] = nu_shock[:, mask_4] / nu_shock.sum(axis=0).A1[mask_4]

        eta_exp_shock[mask, :] = (eta_exp[mask, :].multiply(transition_eta_exp_multi[mask, :]) +
                                  transition_eta_exp_rewire[mask, :]).multiply(rl[mask])

        eta_prod_shock[mask, :] = (eta_prod[mask, :].multiply(transition_eta_prod_multi[mask, :]) +
                                   transition_eta_prod_rewire[mask, :]).multiply(rl[mask])

        eta_cons_shock[mask, :] = (eta_cons[mask, :].multiply(transition_eta_cons_multi[mask, :]) +
                                   transition_eta_cons_rewire[mask, :]).multiply(rl[mask])

        # Compute faktor
        faktor = eta_exp_shock[mask, :] + eta_prod_shock[mask, :] + eta_cons_shock[mask, :]

        
        exp_nonzero = eta_exp_shock[mask, :].nonzero()[1]
        prod_nonzero = eta_prod_shock[mask, :].nonzero()[1]
        cons_nonzero = eta_cons_shock[mask, :].nonzero()[1]

        
        min_length = min(len(exp_nonzero), len(prod_nonzero), len(cons_nonzero), len(faktor.data))
        common_indices = np.arange(min_length)
        valid_mask = (faktor.data[common_indices] > 0) & (~np.isnan(faktor.data[common_indices]))

        
        for i in common_indices[valid_mask]:
            if i < len(exp_nonzero):
                eta_exp_shock.data[exp_nonzero[i]] /= faktor.data[i]
            if i < len(prod_nonzero):
                eta_prod_shock.data[prod_nonzero[i]] /= faktor.data[i]
            if i < len(cons_nonzero):
                eta_cons_shock.data[cons_nonzero[i]] /= faktor.data[i]

        
        for i in range(len(faktor.data)):
            if i >= min_length:
                break
            if faktor.data[i] > 0 and not np.isnan(faktor.data[i]):
                try:
                    if i < len(exp_nonzero):
                        eta_exp_shock.data[exp_nonzero[i]] /= faktor.data[i]
                    if i < len(prod_nonzero):
                        eta_prod_shock.data[prod_nonzero[i]] /= faktor.data[i]
                    if i < len(cons_nonzero):
                        eta_cons_shock.data[cons_nonzero[i]] /= faktor.data[i]
                except IndexError:
                    continue

        
        eta_exp_shock.eliminate_zeros()
        eta_prod_shock.eliminate_zeros()
        eta_cons_shock.eliminate_zeros()

        eta_exp_shock.data = np.nan_to_num(eta_exp_shock.data, nan=0.0)
        eta_prod_shock.data = np.nan_to_num(eta_prod_shock.data, nan=0.0)
        eta_cons_shock.data = np.nan_to_num(eta_cons_shock.data, nan=0.0)

        T_shock[mask, :] = (T[mask, :].multiply(transition_import_multi[mask, :]) + transition_import_rewire[mask, :]).multiply(rl[mask])
        T_shock[:, mask] = (T[:, mask].multiply(transition_export_multi[:, mask]) + transition_export_rewire[:, mask]).multiply(rl[mask].T)
        mask_5 = (T_shock.sum(axis=0).A1 > 0) & ((T_shock.sum(axis=0).A1 < 0.99) | (T_shock.sum(axis=0).A1 > 1.01))
        T_shock[:, mask_5] = T_shock[:, mask_5] / T_shock.sum(axis=0).A1[mask_5]

    if t == 2 and compensation:

        change_rl = set(np.where(rl.toarray()[:, 0] > limit_rel_sim)[0])
        change_al = set(np.where(al.toarray()[:, 0] > limit_abs_sim)[0])
        change = np.array(list(change_rl.intersection(change_al)))
        mask_subs = np.isin(np.arange(Na * Ni), change)

        mask_subs_2 = substitutability_trade[mask_subs].nonzero()[1]

        
        substitutability_factor = np.minimum(substitutability_trade[mask_subs].data + 1, 2.0)
        T_shock[mask_subs_2, :] = T_shock[mask_subs_2, :].multiply(
            sprs.csr_matrix(substitutability_factor).T)

        
        col_sums = T_shock.sum(axis=0).A1
        
        
        zero_cols = col_sums == 0
        if np.any(zero_cols):
            rows = np.where(zero_cols)[0]
            T_shock[rows, rows] = 1.0  # Add 1.0 to diagonal
            col_sums[zero_cols] = 1.0  # Update sums
        
        
        T_shock = T_shock.multiply(1 / col_sums)

        
        final_sums = T_shock.sum(axis=0).A1
        print(f"Normalization check - Min: {final_sums.min():.6f}, Max: {final_sums.max():.6f}")
    # Store
    XS.loc[idx[:, :], 'amount [t]'] = xs.toarray()[:, 0]  # Update final values




if compensation:
    base_filename = f"{scenario}.csv"
else:
    base_filename = f"{scenario}_no_comp.csv"


if production_cap:
    output_filename = base_filename.replace('.csv', '_capped.csv')
else:
    output_filename = base_filename.replace('.csv', '_no_cap.csv')

# Save the results
XS.to_csv(os.path.join(output_folder, output_filename))

print(f'Shocked scenario saved to: {output_filename}')

print("Uncapped production (o):", o.sum())
print("Total supply (xs):", xs.sum())

print(f'Shocked scenario done.')

current_scenario_data = pd.DataFrame(overshoot_data)

# Remove any existing data for this scenario to avoid duplicates
if not all_overshoot_data.empty:
    all_overshoot_data = all_overshoot_data[all_overshoot_data['scenario'] != scenario]

# Append new data
all_overshoot_data = pd.concat([all_overshoot_data, current_scenario_data], ignore_index=True)

# Save the complete dataset
all_overshoot_data.to_csv(overshoot_output_file, index=False)

