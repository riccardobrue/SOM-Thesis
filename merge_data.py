from openpyxl import load_workbook
import pandas as pd
import numpy as np

manipulated_csv_name = 'manipulated_data.csv'


def get_data(restore=False):
    if restore:
        dataframe = pd.read_csv(manipulated_csv_name)
    else:
        # ----------------------------------------------------------
        # LOAD DATA FROM FILE
        # ----------------------------------------------------------
        data_path = "C:\\Users\\Riccardo\\Google Drive\\University\\Double Degree - Middlesex\\Middlesex Teaching Material\\CSD4444 - Ralph Moseley\\Data\\Temporary fake data\\"
        data_file_name = "AREA_50x50.xlsx"
        complete_file_path = data_path + data_file_name

        wb = load_workbook(complete_file_path, data_only=True)  # , read_only=True)

        # selecting the first (or active) sheet
        ws = wb.active
        # or
        # first_sheet = wb.get_sheet_names()[0]
        # ws = wb.get_sheet_by_name(first_sheet)

        columns = np.array(['A', 'B', 'C', 'D', 'E', 'F'])  # is the first output column
        num_attributes = 5

        data_dictionary = {}
        max_rows = 0  # the first column defines the maximum number of rows
        for c in range(len(columns)):
            column = ws[columns[c]]
            column_header = str(column[0].value)
            data_dictionary.update({column_header: []})

            for x in range(1, len(column)):
                if column[x].value is None:
                    if c == 0:
                        max_rows = x
                        break
                    elif c > 0 and x < max_rows:
                        data_dictionary[column_header].append(column[x].value)
                    else:
                        break
                else:
                    data_dictionary[column_header].append(column[x].value)
        dataframe = pd.DataFrame(data=data_dictionary, columns=data_dictionary.keys())

        # print(dataframe)

        """
        # ----------------------------------------------------------
        # MANIPULATING DATA
        # ----------------------------------------------------------
        dataframe['HEED FND'] = dataframe.apply(
            lambda row: row['R0'] * row['HOM ENERGY'],
            # lambda row: row['R0']*row['HOM ENERGY'] if np.isnan(row['c']) else row['c'],
            axis=1
        )
        # print(dataframe)
        """
        # ----------------------------------------------------------
        # STORING DATA TO A SIMPLEST CSV
        # ----------------------------------------------------------
        dataframe.to_csv(manipulated_csv_name, index=False)
    # ----------------------------------------------------------
    # CONVERT DATAFRAME TO NP ARRAY
    # ----------------------------------------------------------
    column_names = list(dataframe.columns.values)
    input_cols = [column_names[0], column_names[1], column_names[2], column_names[3], column_names[4]]
    output_cols = [column_names[5]]

    inputs = dataframe.as_matrix(input_cols)
    efficiency_values = dataframe.as_matrix(output_cols)
    efficiency_values = np.squeeze(efficiency_values)
    efficiency_values = list(efficiency_values)

    return inputs, efficiency_values
