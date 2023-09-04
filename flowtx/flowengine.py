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
from typing import Optional


class FlowEngine:

    wsp: fk.Workspace

    def __init__(self, wsp_path: Optional[str] = None, use_cache: bool = True, cache_path: str = "wsp_cache.pkl"):
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

        self.gating_results = self.wsp.get_analysis_report()

        self.rebuild_df()

    def rebuild_df(self):
        """
        Rebuild the dataframe from the samples.
        """
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

        self.df = pd.DataFrame(flattened)

    def add_sample(self, condition_specifier: dict, well_specifier: dict, data_address: str = None) -> None:
        sample = {}

        sample['condition'] = condition_specifier
        sample['well'] = well_specifier
        sample['data_address'] = data_address

        gate_counts = self.gate_counts_for_sample_data(data_address)

        sample['gate_counts'] = gate_counts

        self.samples.append(sample)

        self.rebuild_df()

    def report(self):
        """
        Report on data. This assumes all the samples have the same gating strategy and channels.

        """
        # Get the first sample
        sample_id = self.wsp.get_sample_ids()[0]
        sample = self.wsp.get_sample(sample_id)
        print('Fluorescence Channels:')
        print('-'*40)
        print(sample.channels['pnn'])

        # Get the gating strategy
        sample_id = self.wsp.get_sample_ids()[0]

        gating_results = self.wsp.get_analysis_report()

        sample_results = gating_results[gating_results['sample'] == sample_id]

        gates = sample_results['gate_name'].unique()

        print('Gates')
        print('-'*40)
        for gate in gates:
            print("Gate: %s" % gate)

        print('Gate Tree')
        print('-'*40)
        print(self.wsp.get_gate_hierarchy(sample_id))

    def gate_counts_for_sample_data(self, data_address: str):

        # Try to find the sample in the FlowJo workspace
        sample_ids = self.wsp.get_sample_ids()  # These are the FlowJo names

        if data_address in sample_ids:

            # sample = self.wsp.get_sample(data_address)

            # Hierarchy for sample
            # print(self.wsp.get_gate_hierarchy(data_address))

            sample_results = self.gating_results[self.gating_results['sample'] == data_address]

            gate_counts = {}

            for gate_name in sample_results['gate_name'].unique():
                gate_counts['count %s' % gate_name] = sample_results[sample_results['gate_name']
                                                                     == gate_name]['count'].values[0]

            return gate_counts
        else:
            print('Sample not found in workspace.')
            return None

    def plot_timecourses(self, rows_field, cols_field, fields_to_plot, time_col, gate_cols_prefix=None, df=None, twinax=True):
        """
        Plot the raw timecourses for each condition.

        Parameters
        ----------
        rows_field : str
            Name of the column in the dataframe that contains the row values.
        cols_field : str
            Name of the column in the dataframe that contains the column values.
        fields_to_plot : list or str
            List of column names in the dataframe to plot.
            Or str column name that contains the gate names to plot.
        time_col : str
            Name of the column in the dataframe that contains the time values.

        """
        if df is None:
            df = self.df

        default_colors = mpl.rcParams['axes.prop_cycle'].by_key()['color']
        default_colors = ['black', 'blue', 'red']

        col_values = df[cols_field].unique()
        row_values = df[rows_field].unique()

        fig, axs = plt.subplots(len(row_values), len(col_values), figsize=(
            3.4*len(col_values), 3.5*len(row_values)), sharey=True)
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

                if twinax:
                    hax = ax.twinx()
                    haxs.append(hax)

                target = df[(df[cols_field] == col_name) & (df[rows_field] == row_name)]

                if isinstance(fields_to_plot, str):
                    gates = target[fields_to_plot].iloc[0]

                    gate_cols = []
                    for gate in gates:
                        if gate_cols_prefix:
                            gate_cols.append('%sgate_counts count %s' % (gate_cols_prefix, gate))
                        else:
                            gate_cols.append('gate_counts count %s' % gate)

                    _fields_to_plot = gate_cols
                    # print(fields_to_plot)

                else:
                    _fields_to_plot = fields_to_plot

                for k, readout_name in enumerate(_fields_to_plot):
                    color = default_colors[k % len(default_colors)]

                    # Plot the second species on the twinned axis.
                    if len(_fields_to_plot) > 1 and k == 1 and twinax:
                        tax = hax
                        tax.spines['right'].set_color(color)
                        tax.yaxis.label.set_color(color)
                        tax.tick_params(axis='y', colors=color)
                    else:
                        tax = ax

                    sns.scatterplot(ax=tax, x=time_col, y=readout_name, data=target, color=color, legend=False)
                    sns.lineplot(ax=tax, x=time_col, y=readout_name, data=target,
                                 label=readout_name, errorbar=None, color=color, legend=False)

                    if j != 0:  # Only the first row gets a base y-axis label
                        ax.set_ylabel('')

                    # Only the last column gets a twinned y-axis label
                    if j != len(col_values) - 1 and twinax:
                        hax.set_ylabel('')

                    if j == 0:
                        if twinax:
                            handles1, labels1 = ax.get_legend_handles_labels()
                            handles2, labels2 = hax.get_legend_handles_labels()

                            all_handles = handles1 + handles2
                            all_labels = labels1 + labels2

                            unique_labels, idx = np.unique(all_labels, return_index=True)
                            unique_handles = [all_handles[i] for i in idx]

                            ax.legend(unique_handles, unique_labels)
                        else:
                            ax.legend()
                    else:
                        tax.legend([], [], frameon=False)
                        ax.legend([], [], frameon=False)

        # After all plotting operations, synchronize y-limits across all twinned axes
        if twinax:
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

    def plot_histograms(self, rows_field, cols_field, channel_name, gate_name, transform=False, wells='A|D'):
        """
        Plot histograms for each condition using raw events.
        """

        default_colors = mpl.rcParams['axes.prop_cycle'].by_key()['color']
        default_colors = ['black', 'blue', 'red']

        col_values = self.df[cols_field].unique()
        row_values = self.df[rows_field].unique()

        fig, axs = plt.subplots(len(row_values), len(col_values), figsize=(
            3.4*len(col_values), 3.5*len(row_values)), sharey=True)
        fig.suptitle('Raw Data Histograms', fontsize=30)

        for j, col_name in enumerate(col_values):
            for i, row_name in enumerate(row_values):
                ax = axs[i, j]
                ax.set_title('%s \n %s' % (col_name, row_name))

                target = self.df[(self.df[cols_field] == col_name) & (self.df[rows_field] ==
                                                                      row_name) & (self.df['well wells'].str.contains(wells))]
                sample_ids = target['data_address'].unique()  # Assuming each target dataframe has unique sample_ids
                timepoints = target['well timepoint'].unique()
                for k, sample_id in enumerate(sample_ids):
                    raw_data = self.get_raw_events_for_sample(sample_id, channel_name, gate_name, transform)

                    color = default_colors[j % len(default_colors)]
                    # sns.histplot(raw_data, ax=ax, color=color, label=row_name)
                    sns.kdeplot(raw_data, ax=ax, log_scale=True, bw_adjust=0.5, label=timepoints[k], fill=True)

                if j != 0:  # Only the first row gets a base y-axis label
                    ax.set_ylabel('')
                if i == 0 and j == 0:
                    ax.legend()
                else:
                    ax.legend([], [], frameon=False)

        plt.tight_layout()
        return fig

    @staticmethod
    def normalize_data(df):

        def compute_normalization_factors(df):
            # Filter for timepoint zero
            timepoint_zero = df[df['well timepoint'] == 0]

            # Compute mean for each combination of `condition effectors` and `condition condition`
            mean_values = timepoint_zero.groupby(['condition effectors', 'condition condition']).mean().reset_index()

            return mean_values

        mean_values = compute_normalization_factors(df)
        gate_counts_columns = [col for col in df.columns if 'gate_counts' in col]

        # Create a copy of df to store normalized values
        normalized_df = df.copy()

        for col in gate_counts_columns:
            new_col_name = f"normalized_{col}"

            normalized_df = pd.merge(normalized_df,
                                     mean_values[['condition effectors', 'condition condition', col]],
                                     on=['condition effectors', 'condition condition'],
                                     how='left',
                                     suffixes=('', '_mean'))

            normalized_df[new_col_name] = normalized_df[col] / normalized_df[f"{col}_mean"]
            normalized_df.drop(f"{col}_mean", axis=1, inplace=True)

        return normalized_df

    def normalize_counts(self):
        """
        Normalize the counts by the counts at timepoint zero.
        """
        self.df = self.normalize_data(self.df)

    def visualize_t0_counts(self, counts_col, df=None):
        """
        Visualize the specified column from the dataframe with thresholds for outlier detection.

        Parameters
        ----------
        df : DataFrame
            The dataframe containing the data to be visualized.
        counts_col : str
            The column name of the data to be plotted on the y-axis.

        Returns
        -------
        fig : Figure
            A matplotlib Figure object containing the generated plot.

        """

        if df is None:
            df = self.df[self.df['well timepoint'] == 0]

        # Create a figure and axis object
        fig, ax = plt.subplots(figsize=(10, 3), dpi=300)

        # Generate the x-values
        x_values = range(len(df))

        # Scatter plot with Seaborn
        sns.scatterplot(x=x_values, y=df[counts_col], hue=df['condition effectors'], ax=ax)

        # Optional styling
        ax.set_xlabel('Sample Index')
        ax.set_ylabel(counts_col)
        ax.legend(title='condition effector', loc='upper left', bbox_to_anchor=(1, 1))

        # Determine quantiles and outlier thresholds
        Q1 = df[counts_col].quantile(0.25)
        Q3 = df[counts_col].quantile(0.75)
        IQR = Q3 - Q1
        lower_threshold = Q1 - 1.5 * IQR
        upper_threshold = Q3 + 1.5 * IQR

        # Draw lines for quantiles and thresholds
        ax.axhline(y=lower_threshold, color='grey', linestyle='--', label='Lower Threshold', alpha=0.5)
        ax.axhline(y=upper_threshold, color='grey', linestyle='--', label='Upper Threshold', alpha=0.5)

        # Add text labels for the thresholds
        ax.text(x_values[-1] + 0.5, lower_threshold, 'Lower Threshold (Q1 - 1.5 * IQR)',
                verticalalignment='bottom', horizontalalignment='right', color='grey')
        ax.text(x_values[-1] + 0.5, upper_threshold, 'Upper Threshold (Q3 + 1.5 * IQR)',
                verticalalignment='bottom', horizontalalignment='right', color='grey')

        for idx, x_val in enumerate(x_values):
            current_value = df.iloc[idx][counts_col]
            if current_value < lower_threshold or current_value > upper_threshold:
                ax.text(x_val, current_value, f"{df.iloc[idx]['well plate']} {df.iloc[idx]['well wells']}")

        plt.tight_layout()

        return fig

    # Sample usage (assuming you have a DataFrame `sample_df`):
    # fig = visualize_t0_plate(sample_df, "some_column_name")
    # fig.savefig("output.png")

    def plot_timecourses_by_condition(self, effectors_list, df=None):
        """Figure for each `condition condition`, trace for each `condition effectors` in `effectors_list

        Parameters
        ----------
        df : df, optional
            _description_
        effectors_list : list[str]
            _description_
        """
        if df is None:
            df = self.df

        df = df[df['condition effectors'].isin(effectors_list)]

        # concatenate all the species_lists and then get the unique values
        total_species_list = np.unique(np.concatenate(df['condition species_list'].to_list()))

        figs = []
        # Plot each condition with each species as a subplot
        for condition in df['condition condition'].unique():
            fig, axs = plt.subplots(1, len(total_species_list), figsize=(len(total_species_list)*4 + 2, 4), dpi=300)
            fig.suptitle(condition)

            tdf = df[df['condition condition'] == condition]

            for i, species in enumerate(total_species_list):
                if species not in tdf['condition species_list'].iloc[0]:
                    continue
                for j, effector_condition in enumerate(tdf['condition effectors'].unique()):
                    sdf = tdf[tdf['condition effectors'] == effector_condition]
                    sns.lineplot(x='well timepoint', y='normalized_gate_counts count %s' % species,
                                 data=sdf, ax=axs[i], label=effector_condition, errorbar='se', err_style='band', marker='o')

                    axs[i].set_title('%s | %s' % (condition, species))
                    axs[i].set_xlabel('Time (hours)')
                    axs[i].set_ylabel('Normalized Counts')

                    # Remove legend for all but the last subplot
                    if i < len(total_species_list) - 1:
                        axs[i].legend().set_visible(False)

            # Add legend to the right of the last subplot
            axs[-1].legend(loc='upper left', bbox_to_anchor=(1, 1))

            plt.tight_layout()
            figs.append(fig)

        return figs

    def plot_timecourses_by_effectors(self, effectors_list, df=None):
        """Figure for each `condition condition`, trace for each `condition effectors` in `effectors_list

        Parameters
        ----------
        df : df, optional
            _description_
        effectors_list : list[str]
            _description_
        """
        if df is None:
            df = self.df

        df = df[df['condition effectors'].isin(effectors_list)]

        # concatenate all the species_lists and then get the unique values
        total_species_list = np.unique(np.concatenate(df['condition species_list'].to_list()))

        figs = []
        # Plot each condition with each species as a subplot
        # Plot for each effector condition with each species as a subplot
        for effector_condition in effectors_list:
            fig, axs = plt.subplots(1, len(total_species_list), figsize=(len(total_species_list)*4 + 2, 4), dpi=300)
            fig.suptitle(effector_condition)

            # Filter by effector condition
            edf = df[df['condition effectors'] == effector_condition]

            for i, species in enumerate(total_species_list):

                for j, condition in enumerate(edf['condition condition'].unique()):
                    cdf = edf[edf['condition condition'] == condition]
                    if species not in cdf['condition species_list'].iloc[0]:
                        continue
                    sns.lineplot(x='well timepoint', y='normalized_gate_counts count %s' % species,
                                 data=cdf, ax=axs[i], label=condition, errorbar='se', err_style='band', marker='o')
                    axs[i].set_title(species)
                    axs[i].set_xlabel('Time (hours)')
                    axs[i].set_ylabel('Normalized Counts')

                    # Remove legend for all but the last subplot
                    if i < len(total_species_list) - 1:
                        axs[i].legend().set_visible(False)

            # Add legend to the right of the last subplot
            axs[-1].legend(loc='upper left', bbox_to_anchor=(1, 1))
            plt.tight_layout()

            figs.append(fig)

        return figs
