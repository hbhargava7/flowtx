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


    @classmethod
    def load_engine_from_file(cls, path: str):
        """Instantiates a FlowEngine object from a serialized record.
        
        Parameters
        ----------
        path : str
            The path from where the serialized FlowEngine object should be loaded.
        
        Returns
        -------
        FlowEngine
            An instance of FlowEngine loaded from the serialized record.
        """
        
        with open(path, 'rb') as file:
            instance = pickle.load(file)
        
        if not isinstance(instance, cls):
            raise TypeError(f"Unserialized object is of type {type(instance)}, expected {cls}.")
        
        return instance

    def save_to_file(self, path: str):
        """Serializes the FlowEngine object and saves it to a specified path.
        
        Parameters
        ----------
        path : str
            The path where the serialized FlowEngine object should be saved.
        """
        
        with open(path, 'wb') as file:
            pickle.dump(self, file)


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
            print('Sample not found in workspace: %s' % data_address)
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
            3.4*len(col_values), 3.5*len(row_values)), sharey=True, sharex=True)
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
                    try:
                        gates = target[fields_to_plot].iloc[0]

                        gate_cols = []
                        for gate in gates:
                            if gate_cols_prefix:
                                gate_cols.append('%sgate_counts count %s' % (gate_cols_prefix, gate))
                            else:
                                gate_cols.append('gate_counts count %s' % gate)

                        _fields_to_plot = gate_cols
                    except Exception as e:
                        print('Error plotting %s | %s' % (row_name, col_name))
                        continue
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

                    # Drop nan values
                    _target = target.dropna(subset=[time_col, readout_name])


                    sns.scatterplot(ax=tax, x=time_col, y=readout_name, data=_target, color=color, legend=False)
                    sns.lineplot(ax=tax, x=time_col, y=readout_name, data=_target,
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

        # is_subsample = np.zeros(sample.event_count, dtype=bool)
        # is_subsample[sample.subsample_indices] = True

        # idx_to_plot = np.logical_and(is_gate_event, is_subsample)

        x = x[is_gate_event]

        return x

    def plot_histograms_for_samples(self, sample_ids, channel_name, gate_name, transform=None, kde_bw=0.1, norm=False, labels=None):
        """Plot histograms for one or more `sample_ids`.

        Parameters
        ----------
        sample_ids : list[str]
            List of sample ids (passed to `self.get_raw_events_for_sample`)
        channel_name : str
            Fluorescence channel
        gate_name : str
            Gate name
        transform : callable, optional
            transforming function, optional (see flowkit transforms)
        kde_bw : float, optional
            bandwidth for KDE smoothing, by default 0.1
        norm : bool
            Normalize the scale of the histograms
        labels : list[str], optional
            Labels in the same order as `sample_ids` for the traces. Otherwise will label with `sample_ids.

        """

        fig, ax = plt.subplots(figsize=(8, 4), dpi=150)

        from scipy.stats import gaussian_kde

        if labels is None:
            labels = sample_ids

        data = []
        if norm == False:
            # Aggregate all the data
            for k, sample_id in enumerate(sample_ids):
                    
                    # Get the raw data for the sample
                    raw_data = self.get_raw_events_for_sample(sample_id, channel_name, gate_name, transform)

                    # Drop negative, inf, or nan values
                    raw_data = raw_data[np.isfinite(raw_data)]  # Removes inf and nan
                    raw_data = raw_data[raw_data > 0]  # Removes negative values

                    # Log transform the data                 
                    raw_data = np.log10(raw_data)
                    data.append(raw_data)

            for l, _data in enumerate(data):

                # plot KDE
                kde = gaussian_kde(_data, bw_method=kde_bw)

                # Get the bounds of the data
                x_grid = np.linspace(min(np.hstack(data)), max(np.hstack(data)*1.2), 1000)
                density = kde.evaluate(x_grid) * len(_data)
                    
                # Plot the KDE
                ax.plot(10**x_grid, density, label=labels[l])
                ax.fill_between(10**x_grid, density, 0, alpha=.25)

            ax.set_xscale('log')

        else:
            
            for k, sample_id in enumerate(sample_ids):
                try:
                    raw_data = self.get_raw_events_for_sample(sample_id, channel_name, gate_name, transform)
                    raw_data = raw_data[np.isfinite(raw_data)]  # Removes inf and nan
                    raw_data = raw_data[raw_data > 0]  # Removes negative values
                    sns.kdeplot(raw_data, ax=ax, log_scale=True, bw_adjust=0.5, label=labels[k], fill=True, common_norm=norm)

                except Exception as e:
                    print('Failed to plot %s: %s' % (sample_id, e))

        ax.set_xlabel(channel_name)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()

        return fig


    def plot_histograms(self, rows_field, cols_field, channel_name, gate_name, transform=False, wells='A|D', norm=False, kde_bw=0.1):
        """
        Plot histograms for each condition using raw events.

        Parameters
        ----------
        rows_field : str

        cols_field : str

        channel_name : str

        gate_name : str

        transform : bool, optional

        wells : str, optional

        norm : bool, optional
            Whether to normalize the histograms.

            if False, will scale by the number of cells.

        Returns
        -------
        fig : Figure
            A matplotlib Figure object containing the generated plot.
        """
        from scipy.stats import gaussian_kde

        default_colors = mpl.rcParams['axes.prop_cycle'].by_key()['color']
        default_colors = ['black', 'blue', 'red']

        col_values = self.df[cols_field].unique()
        row_values = self.df[rows_field].unique()

        fig, axs = plt.subplots(len(row_values), len(col_values), figsize=(
            3.4*len(col_values), 3.5*len(row_values)), sharey=True, sharex=True)
        fig.suptitle('Raw Data Histograms', fontsize=30)

        for j, col_name in enumerate(col_values):
            for i, row_name in enumerate(row_values):
                ax = axs[i, j]
                ax.set_title('%s \n %s' % (col_name, row_name))

                target = self.df[(self.df[cols_field] == col_name) & (self.df[rows_field] ==
                                                                      row_name) & (self.df['well wells'].str.contains(wells))]
                sample_ids = target['data_address'].unique()  # Assuming each target dataframe has unique sample_ids
                timepoints = target['well timepoint'].unique()

                data = []
                if norm == False:
                    # Aggregate all the data
                    for k, sample_id in enumerate(sample_ids):
                            
                            # Get the raw data for the sample
                            raw_data = self.get_raw_events_for_sample(sample_id, channel_name, gate_name, transform)

                            # Drop negative, inf, or nan values
                            raw_data = raw_data[np.isfinite(raw_data)]  # Removes inf and nan
                            raw_data = raw_data[raw_data > 0]  # Removes negative values

                            # Log transform the data
                            raw_data = np.log10(raw_data)
                            data.append(raw_data)

                    for l, _data in enumerate(data):
                        try:
                            # plot KDE
                            kde = gaussian_kde(_data, bw_method=kde_bw)

                            # Get the bounds of the data
                            x_grid = np.linspace(min(np.hstack(data)), max(np.hstack(data)), 1000)
                            density = kde.evaluate(x_grid) * len(_data)
                            
                            # Plot the KDE
                            ax.plot(10**x_grid, density, label=timepoints[l])
                            ax.fill_between(10**x_grid, density, 0, alpha=.25)
                        except Exception as e:
                            print('Failed to plot %s: %s' % (sample_id, e))

                    ax.set_xscale('log')

                else:
                   
                    for k, sample_id in enumerate(sample_ids):
                        try:
                            raw_data = self.get_raw_events_for_sample(sample_id, channel_name, gate_name, transform)

                            sns.kdeplot(raw_data, ax=ax, log_scale=True, bw_adjust=0.5, label=timepoints[k], fill=True, common_norm=norm)

                        except Exception as e:
                            print('Failed to plot %s: %s' % (sample_id, e))

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
            timepoint_zero = df[df['well timepoint'] == np.min(df['well timepoint'])]

            # Compute mean for each combination of `condition effectors` and `condition condition`
            mean_values = timepoint_zero.groupby(['condition effectors', 'condition condition']).mean(numeric_only=True).reset_index()

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

        normalized_df = normalized_df.replace([np.inf, -np.inf], np.nan)
        return normalized_df

    def normalize_counts(self):
        """
        Normalize the counts by the counts at timepoint zero.
        """
        self.df = self.normalize_data(self.df)

    def visualize_raw_counts(self, counts_col, timepoint=None, df=None, iqr_fold=1.5):
        """
        Visualize the specified column from the dataframe with thresholds for outlier detection.

        Parameters
        ----------
        df : DataFrame
            The dataframe containing the data to be visualized.
        counts_col : str
            The column name of the data to be plotted on the y-axis.
        timepoint : int
            The timepoint to be visualized. if `df` not provided we will filter for df['well timepoint'] == timepoint

        Returns
        -------
        fig : Figure
            A matplotlib Figure object containing the generated plot.

        """

        if df is None:
            df = self.df[self.df['well timepoint'] == timepoint]

        # Create a figure and axis object
        fig, ax = plt.subplots(figsize=(10, 3), dpi=300)

        # Generate the x-values
        x_values = range(len(df))
        if timepoint is not None:
            fig.suptitle('Raw Counts (t=%s)' % timepoint, fontsize=24)
        else:
            fig.suptitle('Raw Counts', fontsize=24)
                         
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
        lower_threshold = Q1 - iqr_fold * IQR
        upper_threshold = Q3 + iqr_fold * IQR

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
                outlier_label =f"{df.iloc[idx]['well plate']} {df.iloc[idx]['well wells']}" 
                ax.text(x_val, current_value, outlier_label)
                print("Outlier for %s: %s" % (counts_col, outlier_label))

        plt.tight_layout()

        return fig

    # Sample usage (assuming you have a DataFrame `sample_df`):
    # fig = visualize_t0_plate(sample_df, "some_column_name")
    # fig.savefig("output.png")

    def get_total_species_list(self, df=None):
        """
        Get the total list of species across all conditions.

        Parameters
        ----------
        df : DataFrame
            The dataframe containing the data to be visualized.

        Returns
        -------
        species_list : list
            A list of all the species across all conditions.

        """
        if df is None:
            df = self.df

        # concatenate all the species_lists and then get the unique values
        species_list = np.unique(np.concatenate(df['condition species_list'].to_list()))

        return species_list

    def plot_timecourses_by_condition(self, effectors_list, df=None, title=None):
        """Figure for each `condition condition`, trace for each `condition effectors` in `effectors_list

        Parameters
        ----------
        df : df, optional
            _description_
        effectors_list : list[str]
            _description_
        title : str, optional
            String title for plots
        """
        
        if df is None:
            df = self.df

        df = df[df['condition effectors'].isin(effectors_list)]

        # concatenate all the species_lists and then get the unique values
        total_species_list = np.unique(np.concatenate(df['condition species_list'].to_list()))

        # Retrieve the color cycle from the current matplotlib style
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        unique_effectors = df['condition effectors'].unique()
        color_dict = {effector: color_cycle[i % len(color_cycle)] for i, effector in enumerate(unique_effectors)}

        figs = []
        # Plot each condition with each species as a subplot
        for condition in df['condition condition'].unique():
            fig, axs = plt.subplots(1, len(total_species_list), figsize=(len(total_species_list)*4 + 2, 4), dpi=300)

            if title:
                fig.suptitle('%s\nCondition: %s' % (title, condition), fontsize=25, fontweight='bold')

            else:
                fig.suptitle(condition, fontsize=25, fontweight='bold')

            tdf = df[df['condition condition'] == condition]

            for i, species in enumerate(total_species_list):
                try:
                    if species not in tdf['condition species_list'].iloc[0]:
                        continue
                    for j, effector_condition in enumerate(tdf['condition effectors'].unique()):
                        sdf = tdf[tdf['condition effectors'] == effector_condition]

                        # Drop any nan values before plotting
                        sdf = sdf.dropna(subset=['well timepoint', 'normalized_gate_counts count %s' % species])

                        sns.lineplot(x='well timepoint', y='normalized_gate_counts count %s' % species,
                                    data=sdf, ax=axs[i], label=effector_condition, errorbar='se', err_style='band', marker='o',
                                    color=color_dict[effector_condition])  # Use the specific color for the current effector_condition


                        axs[i].set_title('%s \n %s' % (condition, species))
                        axs[i].set_xlabel('Time (hours)')
                        axs[i].set_ylabel('Normalized Counts')

                        # Remove legend for all but the last subplot
                        if i < len(total_species_list) - 1:
                            axs[i].legend().set_visible(False)
                except Exception as e:
                    print('Error plotting %s: %s' % (species, e))


            # After plotting all lines in the rightmost subplot, retrieve the lines and labels
            lines, labels = axs[0].get_legend_handles_labels()

            try:
                # Sort the lines and labels based on the y-data of the lines
                sorted_labels, sorted_lines = zip(*sorted(zip(labels, lines), key=lambda t: t[1].get_ydata()[-1], reverse=True))

                # Set the legend with the sorted labels
                axs[-1].legend(sorted_lines, sorted_labels, loc='upper left', bbox_to_anchor=(1, 1))
            except Exception as e:
                print('Error sorting labels: %s' % e)
                axs[-1].legend(loc='upper left', bbox_to_anchor=(1, 1))

            # Add legend to the right of the last subplot
            # axs[-1].legend(loc='upper left', bbox_to_anchor=(1, 1))

            plt.tight_layout()
            figs.append(fig)

        return figs

    def plot_timecourses_by_effectors(self, effectors_list, df=None, title=None):
        """Figure for each `condition condition`, trace for each `condition effectors` in `effectors_list

        Parameters
        ----------
        df : df, optional
            _description_
        effectors_list : list[str]
            _description_
        title : str, optional
            String title for plots
        """
        
        if df is None:
            df = self.df

        df = df[df['condition effectors'].isin(effectors_list)]

        # concatenate all the species_lists and then get the unique values
        total_species_list = np.unique(np.concatenate(df['condition species_list'].to_list()))

        # Retrieve the color cycle from the current matplotlib style
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        unique_conditions = df['condition condition'].unique()
        color_dict = {condition: color_cycle[i % len(color_cycle)] for i, condition in enumerate(unique_conditions)}

        figs = []
        # Plot each condition with each species as a subplot
        # Plot for each effector condition with each species as a subplot
        for effector_condition in effectors_list:
            fig, axs = plt.subplots(1, len(total_species_list), figsize=(len(total_species_list)*4 + 2, 4), dpi=300)
            if title:
                fig.suptitle('%s\nCondition: %s' % (title, effector_condition), fontsize=25, fontweight='bold')
            else:
                fig.suptitle(effector_condition, fontsize=25, fontweight='bold')

            # Filter by effector condition
            edf = df[df['condition effectors'] == effector_condition]

            for i, species in enumerate(total_species_list):
                for j, condition in enumerate(unique_conditions):
                    cdf = edf[edf['condition condition'] == condition]
                    if len(cdf['condition species_list']) == 0:
                        continue

                    if species not in cdf['condition species_list'].iloc[0]:
                        continue

                    cdf = cdf.dropna(subset=['well timepoint', 'normalized_gate_counts count %s' % species])

                    sns.lineplot(x='well timepoint', y='normalized_gate_counts count %s' % species,
                         data=cdf, ax=axs[i], label=condition, errorbar='se', err_style='band', marker='o',
                         color=color_dict[condition])  # Use the specific color for the current condition

                    axs[i].set_title(species)
                    axs[i].set_xlabel('Time (hours)')
                    axs[i].set_ylabel('Normalized Counts')

                    # Remove legend for all but the last subplot
                    if i < len(total_species_list) - 1:
                        axs[i].legend().set_visible(False)
            # After plotting all lines in the rightmost subplot, retrieve the lines and labels
            lines, labels = axs[0].get_legend_handles_labels()

            try:
                # Sort the lines and labels based on the y-data of the lines
                sorted_labels, sorted_lines = zip(*sorted(zip(labels, lines), key=lambda t: t[1].get_ydata()[-1], reverse=True))

                # Set the legend with the sorted labels
                axs[-1].legend(sorted_lines, sorted_labels, loc='upper left', bbox_to_anchor=(1, 1))
            except Exception as e:
                print('Error sorting labels: %s' % e)
                axs[-1].legend(loc='upper left', bbox_to_anchor=(1, 1))

            plt.tight_layout()

            figs.append(fig)

        return figs
