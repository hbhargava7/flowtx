# Copyright 2023 Hersh K. Bhargava (https://hershbhargava.com)
# University of California, San Francisco


import flowkit as fk
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

class Engine():
    def __init__(self, wsp_path: str=None, use_cache: bool=True, cache_path: str="wsp_cache.pkl"):
        """Main class for flow data handling

        Parameters
        ----------
        wsp_path : str, optional
            Path to FlowJo workspace, by default None
            All of the samples to be analyzed must be in this workspace.
        use_cache : bool, optional
            Flag to determine if the cached version of the wsp object should be used, by default False.
        cache_path : str, optional
            Path to the cached wsp object, by default "wsp_cache.pkl".

        """
        self.samples = []

        if use_cache and os.path.exists(cache_path):
            print('Loading cached FlowKit Workspace (wsp) object...')
            with open(cache_path, 'rb') as f:
                self.wsp = pickle.load(f)
            print('Loaded wsp object from cache.')
        else:
            if wsp_path is not None:
                print('Reading FlowJo workspace...')
                self.wsp = fk.Workspace(wsp_path, find_fcs_files_from_wsp=True)

                print('Loaded %i samples from the FlowJo workspace file.' % len(self.wsp.get_sample_ids()))

                print('Executing analysis on workspace with gating strategy from file...')
                self.wsp.analyze_samples()

                print('Done analyzing FCS files.')
                print('Caching wsp object for future use...')
                with open(cache_path, 'wb') as f:
                    pickle.dump(self.wsp, f)
                print('Cached the FlowKit Workspace (wsp) object.')

    @property
    def df(self):
        # Comprehend dicts into columns
        flattened = []
        for sample in self.samples:
            s = {}
            for key in sample.keys():
                if isinstance(sample[key], dict):
                    for k in sample[key].keys():
                        s['%s %s' % (key, k)] = sample[key][k]
                else:
                    s[key] = sample[key]

            flattened.append(s)
        
        return pd.DataFrame(flattened)


    def add_sample(self, condition_specifier: dict, well_specifier: dict, data_address: str = None) -> None:
        sample = {}

        sample['condition'] = condition_specifier
        sample['well'] = well_specifier
        sample['data_address'] = data_address

        gate_counts = self.gate_counts_for_sample_data(data_address)

        sample['gate_counts'] = gate_counts

        self.samples.append(sample)

    def gate_counts_for_sample_data(self, data_address: str):

        # Try to find the sample in the FlowJo workspace
        sample_ids = self.wsp.get_sample_ids() # These are the FlowJo names
 
        if data_address in sample_ids:
            
            sample = self.wsp.get_sample(data_address)

            # Hierarchy for sample
            # print(self.wsp.get_gate_hierarchy(data_address))

            gating_results = self.wsp.get_analysis_report()

            sample_results = gating_results[gating_results['sample'] == data_address]

            gate_counts = {}

            for gate_name in sample_results['gate_name'].unique():
                gate_counts['count %s' % gate_name] = sample_results[sample_results['gate_name'] == gate_name]['count'].values[0]

            return gate_counts
        else:
            print('Sample not found in workspace.')
            return None

    def plot_timecourses(self, rows_field, cols_field, fields_to_plot, time_col):
        """
        Plot the raw timecourses for each condition.

        """

        default_colors = mpl.rcParams['axes.prop_cycle'].by_key()['color']
        default_colors = ['black', 'blue', 'red']

        col_values = self.df[cols_field].unique()
        row_values = self.df[rows_field].unique()

        fig, axs = plt.subplots(len(row_values), len(col_values), figsize=(3.4*len(col_values), 3.5*len(row_values)), sharey=True)
        fig.suptitle('Raw Data', fontsize=30)

        haxs = []

        def get_global_ylims():
            """Get the global y-limits across all twinned axes."""
            all_ylims = [hax.get_ylim() for hax in haxs]
            global_min = min([ylim[0] for ylim in all_ylims])
            global_max = max([ylim[1] for ylim in all_ylims])
            return (global_min, global_max)

        def set_shared_ylims():
            """Sets the y-limits of all twinned axes to the global y-limits."""
            ylims = get_global_ylims()
            for h in haxs:
                h.set_ylim(ylims)

        for j, col_name in enumerate(col_values):
            for i, row_name in enumerate(row_values):
                ax = axs[i, j]
                ax.set_title('%s \n %s' % (col_name, row_name))
                
                hax = ax.twinx()
                haxs.append(hax)

                target = self.df[(self.df[cols_field] == col_name) & (self.df[rows_field] == row_name)]

                for k, readout_name in enumerate(fields_to_plot):
                    color = default_colors[k % len(default_colors)]

                    if len(fields_to_plot) > 1 and k == len(fields_to_plot) - 1:
                        tax = hax
                        tax.spines['right'].set_color(color)
                        tax.yaxis.label.set_color(color)
                        tax.tick_params(axis='y', colors=color)
                    else:
                        tax = ax

                    sns.scatterplot(ax=tax, x=time_col, y=readout_name, data=target, color=color, legend=False)
                    sns.lineplot(ax=tax, x=time_col, y=readout_name, data=target, label=readout_name, ci=None, color=color, legend=False)

                    if j != 0:  # Only the first row gets a base y-axis label
                        ax.set_ylabel('')

                    # Only the last column gets a twinned y-axis label
                    if j != len(col_values) - 1:
                        hax.set_ylabel('')
                        
                    if i == 0 and j == 0:
                        handles1, labels1 = ax.get_legend_handles_labels()
                        handles2, labels2 = hax.get_legend_handles_labels()

                        all_handles = handles1 + handles2
                        all_labels = labels1 + labels2

                        unique_labels, idx = np.unique(all_labels, return_index=True)
                        unique_handles = [all_handles[i] for i in idx]

                        ax.legend(unique_handles, unique_labels)
                    else:
                        tax.legend([],[], frameon=False)
                        ax.legend([],[], frameon=False)

        # After all plotting operations, synchronize y-limits across all twinned axes
        set_shared_ylims()
        plt.tight_layout()

        return fig

    def get_raw_events_for_sample(self, sample_id, channel_name, gate_name, transform=False):
        sample = self.wsp.get_sample(sample_id)
        channel_index = sample.get_channel_index(channel_name)
        x = sample.get_channel_events(channel_index, source='raw', subsample=False)

        # for transform
        if transform:
            x_transform = self.wsp.get_transform(sample_id, channel_name)
            x = x_transform.apply(x)

        gate_results = self.wsp.get_gating_results(sample_id=sample_id)
        is_gate_event = gate_results.get_gate_membership(gate_name)

        is_subsample = np.zeros(sample.event_count, dtype=bool)
        is_subsample[sample.subsample_indices] = True

        idx_to_plot = np.logical_and(is_gate_event, is_subsample)

        x = x[idx_to_plot]

        return x
