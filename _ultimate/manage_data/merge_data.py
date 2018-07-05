import pandas as pd
import numpy as np
import os, os.path
import glob

# Basic path (where the data is stored)
basic_path_desktop = "C:\\Users\\Riccardo\\Google Drive\\University\\Double Degree - Middlesex\\Middlesex Teaching Material\\CSD4444 - Ralph Moseley\\Data\\"
basic_path_laptop = "C:\\Users\\brux9\\Google Drive\\University\\Double Degree - Middlesex\\Middlesex Teaching Material\\CSD4444 - Ralph Moseley\\Data\\"

if os.path.isdir(basic_path_desktop):
    print("DESKTOP")
    basic_path = basic_path_desktop
else:
    print("LAPTOP")
    basic_path = basic_path_laptop

# Simulation folders
simulation_equal_folder_path = basic_path + "Real data\\SimulationEQUAL\\"
simulation_unequal_folder_path = basic_path + "Real data\\SimulationUNEQUAL\\"
# Output folder
output_equal_folder_path = basic_path + "Manipulated data\\Equal\\"
output_unequal_folder_path = basic_path + "Manipulated data\\Unequal\\"
# Output merged single file
output_merged_equal_csv_pathname = output_equal_folder_path + "Merged\\all_merged_equal.csv"
output_merged_unequal_csv_pathname = output_unequal_folder_path + "Merged\\all_merged_unequal.csv"

heights = ["50", "100", "150"]
widths = ["50", "100", "150"]


def create_csv_data(restore=False, selected_height="50", selected_width="50"):
    # output csv (merged) file
    output_merged_equal_file_path = output_equal_folder_path + "merged_equal_" + selected_height + "x" + selected_width + ".csv"
    output_merged_unequal_file_path = output_unequal_folder_path + "merged_unequal_" + selected_height + "x" + selected_width + ".csv"

    if restore:
        # ----------------------------------------------------------
        # READ ALREADY EXTRACTED DATA FROM CSV
        # ----------------------------------------------------------
        dataframe_equal = pd.read_csv(output_merged_equal_file_path)
        dataframe_unequal = pd.read_csv(output_merged_unequal_file_path)
    else:
        # ----------------------------------------------------------
        # CREATE THE CSVs
        # ----------------------------------------------------------
        # --------------------
        # EQUAL PROTOCOLS
        # --------------------
        file_list_equal = os.listdir(simulation_equal_folder_path)  # dir is your directory path

        dictionary_equal = {'HEIGHT': [], 'WIDTH': [], 'NODE': [], 'R0': [], '%AGGR': [], 'HET': [], 'HOM ENERGY': [],
                            'HOM RATE': [], 'REECHD FND': [], 'REECHD HND': [], 'HEED FND': [], 'HEED HND': [],
                            'ERHEED FND': [], 'ERHEED HND': [], 'FMUC FND': [], 'FMUC HND': []}

        for i in range(0, len(file_list_equal)):
            current_file_name = file_list_equal[i]

            width, height, hom_energy, hom_rate, perc_aggr, heterogeneity, transm_range_e = current_file_name.split("_")
            """
            width:              W...        [1:] 
            height:             H...        [1:] 
            hom_energy:         Henergy...  [7:] 
            hom_rate:           Hrate...    [5:]
            perc_aggr:          Agg...      [3:]
            heterogeneity:      Hperc...    [5:]
            transm_range_e:     R0...       [2:]
            """

            width = width[1:]
            height = height[1:]
            hom_energy = hom_energy[7:]
            hom_rate = hom_rate[5:]
            perc_aggr = perc_aggr[3:]
            if perc_aggr == "100percentAggregation":
                perc_aggr = 1
            elif perc_aggr == "70percentAggregation":
                perc_aggr = 0.7
            elif perc_aggr == "40percentAggregation":
                perc_aggr = 0.4
            elif perc_aggr == "NoAggregation":
                perc_aggr = 0
            heterogeneity = heterogeneity[5:]
            transm_range, extension = str(transm_range_e[2:]).split(".")

            if width == selected_width and height == selected_height:
                """
                print(current_file_name, " - ", hom_energy, ", ", hom_rate, ", ", perc_aggr, ", ", heterogeneity, ", ",
                      transm_range)
                """

                dictionary_equal.get("HEIGHT").append(selected_height)
                dictionary_equal.get("WIDTH").append(selected_width)
                dictionary_equal.get("NODE").append(100)

                dictionary_equal.get("R0").append(transm_range)
                dictionary_equal.get("%AGGR").append(perc_aggr)
                dictionary_equal.get("HET").append(heterogeneity)
                dictionary_equal.get("HOM ENERGY").append(hom_energy)
                dictionary_equal.get("HOM RATE").append(hom_rate)

                simulation_dataframe = pd.read_csv(simulation_equal_folder_path + current_file_name, header=None)

                REECHD_FND, REECHD_HND = str(simulation_dataframe.iat[1, 0]).split(";")
                HEED_FND, HEED_HND = str(simulation_dataframe.iat[3, 0]).split(";")
                ERHEED_FND, ERHEED_HND = str(simulation_dataframe.iat[5, 0]).split(";")
                FMUC_FND, FMUC_HND = str(simulation_dataframe.iat[7, 0]).split(";")

                dictionary_equal.get("REECHD FND").append(REECHD_FND)
                dictionary_equal.get("REECHD HND").append(REECHD_HND)
                dictionary_equal.get("HEED FND").append(HEED_FND)
                dictionary_equal.get("HEED HND").append(HEED_HND)
                dictionary_equal.get("ERHEED FND").append(ERHEED_FND)
                dictionary_equal.get("ERHEED HND").append(ERHEED_HND)
                dictionary_equal.get("FMUC FND").append(FMUC_FND)
                dictionary_equal.get("FMUC HND").append(FMUC_HND)

        # --------------------
        # UNEQUAL PROTOCOLS
        # --------------------
        file_list_unequal = os.listdir(simulation_unequal_folder_path)  # dir is your directory path

        dictionary_unequal = {'HEIGHT': [], 'WIDTH': [], 'NODE': [], 'R0': [], '%AGGR': [], 'HET': [], 'HOM ENERGY': [],
                              'HOM RATE': [], 'CONTROL': [], 'UHEED FND': [], 'UHEED HND': [], 'RUHEED FND': [],
                              'RUHEED HND': []}

        for i in range(0, len(file_list_unequal)):
            current_file_name = file_list_unequal[i]

            width, height, control, hom_energy, hom_rate, perc_aggr, heterogeneity, transm_range_e = current_file_name.split(
                "_")

            """
            width:              W...        [1:] 
            height:             H...        [1:] 
            control             Control...  [7:]
            hom_energy:         Henergy...  [7:] 
            hom_rate:           Hrate...    [5:]
            perc_aggr:          Agg...      [3:]
            heterogeneity:      Hperc...    [5:]
            transm_range_e:     R0...       [2:]
            """

            width = width[1:]
            height = height[1:]
            control = control[7:]
            hom_energy = hom_energy[7:]
            hom_rate = hom_rate[5:]
            perc_aggr = perc_aggr[3:]
            if perc_aggr == "100percentAggregation":
                perc_aggr = 1
            elif perc_aggr == "70percentAggregation":
                perc_aggr = 0.7
            elif perc_aggr == "40percentAggregation":
                perc_aggr = 0.4
            elif perc_aggr == "NoAggregation":
                perc_aggr = 0
            heterogeneity = heterogeneity[5:]
            transm_range, extension = str(transm_range_e[2:]).split(".")

            if width == selected_width and height == selected_height:
                """
                print(current_file_name, " - ", control, " - ", hom_energy, ", ", hom_rate, ", ", perc_aggr, 
                    ", ", heterogeneity, ", ",transm_range)
                """

                dictionary_unequal.get("HEIGHT").append(selected_height)
                dictionary_unequal.get("WIDTH").append(selected_width)
                dictionary_unequal.get("NODE").append(100)

                dictionary_unequal.get("R0").append(transm_range)
                dictionary_unequal.get("%AGGR").append(perc_aggr)
                dictionary_unequal.get("HET").append(heterogeneity)
                dictionary_unequal.get("HOM ENERGY").append(hom_energy)
                dictionary_unequal.get("HOM RATE").append(hom_rate)
                dictionary_unequal.get("CONTROL").append(control)

                simulation_dataframe = pd.read_csv(simulation_unequal_folder_path + current_file_name, header=None)

                UHEED_FND, UHEED_HND = str(simulation_dataframe.iat[1, 0]).split(";")
                RUHEED_FND, RUHEED_HND = str(simulation_dataframe.iat[3, 0]).split(";")

                dictionary_unequal.get("UHEED FND").append(UHEED_FND)
                dictionary_unequal.get("UHEED HND").append(UHEED_HND)
                dictionary_unequal.get("RUHEED FND").append(RUHEED_FND)
                dictionary_unequal.get("RUHEED HND").append(RUHEED_HND)

        dataframe_equal = pd.DataFrame(data=dictionary_equal, columns=dictionary_equal.keys())
        dataframe_unequal = pd.DataFrame(data=dictionary_unequal, columns=dictionary_unequal.keys())

        # ----------------------------------------------------------
        # STORING DATA TO A SIMPLEST CSV
        # ----------------------------------------------------------

        dataframe_equal.to_csv(output_merged_equal_file_path, index=False)
        print(dataframe_equal)
        print("---------------------------------------------------------")
        dataframe_unequal.to_csv(output_merged_unequal_file_path, index=False)
        print(dataframe_unequal)


def merge_csv_files():
    all_files_equal = glob.glob(os.path.join(output_equal_folder_path, "*.csv"))
    all_files_unequal = glob.glob(os.path.join(output_unequal_folder_path, "*.csv"))

    df_from_each_file_equal = (pd.read_csv(f) for f in all_files_equal)
    df_from_each_file_unequal = (pd.read_csv(f) for f in all_files_unequal)

    concatenated_df_equal = pd.concat(df_from_each_file_equal, ignore_index=True)
    concatenated_df_unequal = pd.concat(df_from_each_file_unequal, ignore_index=True)

    concatenated_df_equal.to_csv(output_merged_equal_csv_pathname, index=False)
    concatenated_df_unequal.to_csv(output_merged_unequal_csv_pathname, index=False)


def load_equal_data():
    dataframe_equal = pd.read_csv(output_merged_equal_csv_pathname, header=None)

    column_names_equal = list(dataframe_equal.columns.values)

    network_topology_cols_equal = [column_names_equal[0], column_names_equal[1], column_names_equal[2],
                                   column_names_equal[3], column_names_equal[4],
                                   column_names_equal[5], column_names_equal[6], column_names_equal[7]]
    simulation_results_cols_equal = [column_names_equal[8], column_names_equal[9], column_names_equal[10],
                                     column_names_equal[11], column_names_equal[12],
                                     column_names_equal[13], column_names_equal[14], column_names_equal[15]]

    network_topology_att_equal = dataframe_equal.as_matrix(network_topology_cols_equal)[1:]
    simulation_results_equal = dataframe_equal.as_matrix(simulation_results_cols_equal)[1:]
    nt_headers_equal = np.squeeze(dataframe_equal.as_matrix(network_topology_cols_equal)[:1])
    sim_headers_equal = np.squeeze(dataframe_equal.as_matrix(simulation_results_cols_equal)[:1])

    return network_topology_att_equal.astype(float), simulation_results_equal.astype(
        float), nt_headers_equal, sim_headers_equal


def load_unequal_data():
    dataframe_unequal = pd.read_csv(output_merged_unequal_csv_pathname, header=None)

    column_names_unequal = list(dataframe_unequal.columns.values)

    network_topology_cols_unequal = [column_names_unequal[0], column_names_unequal[1], column_names_unequal[2],
                                     column_names_unequal[3], column_names_unequal[4],
                                     column_names_unequal[5], column_names_unequal[6], column_names_unequal[7],
                                     column_names_unequal[8]]
    simulation_results_cols_unequal = [column_names_unequal[9], column_names_unequal[10], column_names_unequal[11],
                                       column_names_unequal[12]]

    network_topology_att_unequal = dataframe_unequal.as_matrix(network_topology_cols_unequal)[1:]
    simulation_results_unequal = dataframe_unequal.as_matrix(simulation_results_cols_unequal)[1:]
    nt_headers_unequal = np.squeeze(dataframe_unequal.as_matrix(network_topology_cols_unequal)[:1])
    sim_headers_unequal = np.squeeze(dataframe_unequal.as_matrix(simulation_results_cols_unequal)[:1])

    return network_topology_att_unequal.astype(float), simulation_results_unequal.astype(
        float), nt_headers_unequal, sim_headers_unequal


if __name__ == "__main__":
    for h in heights:
        for w in widths:
            create_csv_data(selected_height=h, selected_width=w)
    merge_csv_files()
