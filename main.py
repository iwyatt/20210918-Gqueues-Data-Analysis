# -*- coding: utf-8 -*-
"""
Refactored script for GQueues Data Analysis.

This script reads a GQueues task export CSV, processes completed tasks,
analyzes them by tags, and generates a stacked bar chart of tasks
completed per week, broken down by primary tag.
"""

# Import Libraries
import csv
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

# --- Configuration Constants ---
DEFAULT_CSV_PATH = Path("gqueues_backup_2025_05_01.csv")

RELEVANT_DATE_START = pd.Timestamp('2025-01-01')

OUTPUT_CHART_PATH = Path("bar_graph.png")
CSV_ENCODING = "utf-8"
ITEMS_MARKER = "*GQ* Items"
ASSIGNMENTS_MARKER = "*GQ* Assignments"
# --- End Configuration Constants ---


def find_data_section_indices(csv_filepath: Path) -> Tuple[int, int]:
    """
    Scans the CSV to find the header row index and number of data block rows.
    The data block includes the header row.

    Args:
        csv_filepath: Path to the GQueues CSV export.

    Returns:
        A tuple containing:
            - header_row_index (int): 0-indexed row number of the header line for the task data.
            - num_total_data_block_rows (int): Number of rows in the data block, including the header.
                                               This is used for pandas' nrows calculation.
    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If markers for data section are not found or data section is invalid.
    """
    if not csv_filepath.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_filepath}")

    with csv_filepath.open("r", encoding=CSV_ENCODING, newline='') as f:
        reader = csv.reader(f)
        lines = list(reader) 

    header_row_idx = -1
    assignments_marker_idx = -1

    for i, line_content in enumerate(lines):
        # Ensure line_content is not empty and contains strings
        if line_content: # Check if the list of strings is not empty
            # Check if the exact marker string is an element in the list of cells for this row
            if header_row_idx == -1 and ITEMS_MARKER in line_content:
                header_row_idx = i + 1  # Header is the line *after* the marker line
            if ASSIGNMENTS_MARKER in line_content:
                assignments_marker_idx = i
                if header_row_idx != -1:
                    break
    
    if header_row_idx == -1:
        raise ValueError(f"'{ITEMS_MARKER}' marker not found in CSV: {csv_filepath}")
    if assignments_marker_idx == -1:
        raise ValueError(f"'{ASSIGNMENTS_MARKER}' marker not found in CSV: {csv_filepath}")

    last_data_row_idx = assignments_marker_idx - 2

    if last_data_row_idx < header_row_idx:
        raise ValueError(
            "Data section markers indicate an invalid range "
            f"(last data row index {last_data_row_idx} < header row index {header_row_idx}). "
            "Check CSV file structure or marker definitions."
        )

    # Number of rows in the data block, including its header.
    # Block rows = 20 - 10 + 1 = 11 rows.
    num_total_data_block_rows = (last_data_row_idx - header_row_idx) + 1
    
    if num_total_data_block_rows <= 0: # Should be at least 1 (header only)
         raise ValueError(
            f"Calculated number of data block rows ({num_total_data_block_rows}) is not positive. "
            "Check CSV file structure or marker definitions."
        )

    return header_row_idx, num_total_data_block_rows


def load_and_preprocess_tasks(
    csv_filepath: Path, header_row_index: int, num_total_data_block_rows: int
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Loads task data from the specified section of the CSV file,
    converts date columns, and processes tags.

    Args:
        csv_filepath: Path to the GQueues CSV export.
        header_row_index: 0-indexed row number where the data's header is located.
        num_total_data_block_rows: Total number of rows in this data block, including its header.

    Returns:
        A tuple containing:
            - DataFrame: Processed task data.
            - list[str]: Sorted list of unique primary tag names found.
    """
    # `nrows` for pandas is the number of data lines *after* the header.
    # If num_total_data_block_rows is 1 (only header), nrows should be 0.
    num_data_lines_after_header = num_total_data_block_rows - 1

    df = pd.read_csv(
        csv_filepath,
        skiprows=header_row_index,
        nrows=num_data_lines_after_header if num_data_lines_after_header >= 0 else 0,
        encoding=CSV_ENCODING,
        engine='python'
    )

    # Convert 'dateCompleted' from string to datetime objects.
    # Removed explicit format to allow pandas to infer, as input might contain time.
    df["dateCompleted"] = pd.to_datetime(df["dateCompleted"])

    # Process tags:
    # 1. Create a 'primary_tag' column: take the first tag if multiple exist.
    #    Handles np.nan by converting to string 'nan', then 'nan'.split(',')[0] -> 'nan'.
    df['primary_tag'] = df['tags'].apply(
        lambda x: str(x).split(',')[0] if pd.notna(x) else 'nan'
    )
    
    # 2. Get unique primary tags. These will be used as column names.
    unique_primary_tags = sorted(df['primary_tag'].unique().tolist())

    # 3. Create one-shot encoded columns for each unique primary tag.
    for tag_name in unique_primary_tags:
        # Ensure tag_name is a string, as it becomes a column name.
        # unique_primary_tags should already contain strings.
        df[str(tag_name)] = (df['primary_tag'] == tag_name).astype(int)

    return df, unique_primary_tags


def filter_aggregate_and_prepare_chart_data(
    tasks_df: pd.DataFrame,
    min_completion_date: pd.Timestamp,
    tag_column_names: List[str]
) -> pd.DataFrame:
    """
    Filters tasks by completion date, aggregates them by the start of the week,
    and prepares the DataFrame for plotting.

    Args:
        tasks_df: DataFrame containing processed task data.
        min_completion_date: Timestamp, tasks completed on or after this date are included.
        tag_column_names: List of column names representing the one-hot encoded tags.

    Returns:
        DataFrame: Data aggregated by week, ready for plotting.
                   Columns include a date column (week start) and tag count columns.
    """
    if tasks_df.empty: # Handle empty input DataFrame
        # Return an empty DataFrame with expected columns for chart generation to handle gracefully
        return pd.DataFrame(columns=['dateCompleted_week'] + sorted(tag_column_names))

    # Filter DataFrame to the relevant time period
    df_filtered = tasks_df[tasks_df['dateCompleted'] >= min_completion_date].copy()

    if df_filtered.empty: # Handle case where filter results in no data
        return pd.DataFrame(columns=['dateCompleted_week'] + sorted(tag_column_names))

    # Convert 'dateCompleted' to the start date of its week
    df_filtered["dateCompleted_week_dt"] = pd.to_datetime(df_filtered["dateCompleted"])
    df_filtered["dateCompleted_week_dt"] = (
        df_filtered["dateCompleted_week_dt"] -
        pd.to_timedelta(df_filtered["dateCompleted_week_dt"].dt.weekday, unit='D')
    )

    # Format week start date to Python datetime.date object for grouping (strips time)
    df_filtered["dateCompleted_week"] = df_filtered["dateCompleted_week_dt"].dt.date

    # Build aggregated DataFrame: group by week and sum tag counts
    existing_tag_columns = [tag for tag in tag_column_names if tag in df_filtered.columns]
    
    if not existing_tag_columns:
        if 'dateCompleted_week' in df_filtered.columns:
             return df_filtered[['dateCompleted_week']].drop_duplicates().sort_values('dateCompleted_week')
        return pd.DataFrame(columns=['dateCompleted_week'])

    df_aggregated = df_filtered.groupby("dateCompleted_week")[existing_tag_columns].sum()
    df_aggregated = df_aggregated.sort_index().reset_index()

    # Prepare final DataFrame for the graph
    final_graph_columns = ['dateCompleted_week'] + sorted(existing_tag_columns)
    graph_df = df_aggregated[final_graph_columns]
    
    return graph_df


def create_and_save_bar_chart(
    chart_data_df: pd.DataFrame,
    date_column_name: str,
    output_filepath: Path
):
    """
    Creates a stacked bar chart of tasks completed per week by tag and saves it.

    Args:
        chart_data_df: DataFrame prepared for plotting.
        date_column_name: Name of the column in chart_data_df to use for the x-axis.
        output_filepath: Path to save the generated chart image.
    """
    tag_names_to_plot = [col for col in chart_data_df.columns if col != date_column_name]

    if chart_data_df.empty or not tag_names_to_plot:
        print(f"No data available to plot. Skipping chart generation for {output_filepath.name}.")
        fig, ax = plt.subplots(figsize=(18, 12)) # Match figsize for consistency
        ax.text(0.5, 0.5, "No data to display",
                horizontalalignment='center', verticalalignment='center',
                fontsize=20, transform=ax.transAxes)
        ax.set_xlabel("Week Starting On")
        ax.set_ylabel("Number of Tasks Completed")
        ax.set_title("Weekly Task Completion by Tag", fontsize=16)
        fig.savefig(output_filepath, format="png", dpi=150)
        plt.close(fig)
        print(f"Empty chart saved to {output_filepath.resolve()}")
        return

    ax = chart_data_df.plot.bar(
        x=date_column_name,
        y=tag_names_to_plot, # These should be sorted if prepared correctly by previous step
        stacked=True,
        figsize=(18, 12),
        fontsize=10,
        cmap="tab20",
        rot=45,
        xlabel="Week Starting On",
        ylabel="Number of Tasks Completed"
    )
    
    ax.set_title("Weekly Task Completion by Tag", fontsize=16)
    ax.legend(loc="upper left", title="Tags", fontsize=8)
    
    plt.tight_layout(pad=1.5)
    ax.figure.savefig(output_filepath, format="png", dpi=300)
    plt.close(ax.figure)
    
    print(f"Bar chart saved to {output_filepath.resolve()}")


def main():
    """
    Main function to orchestrate the GQueues data analysis.
    """
    print(f"Starting GQueues data analysis for: {DEFAULT_CSV_PATH.name}")
    try:
        header_row_idx, num_block_rows = find_data_section_indices(DEFAULT_CSV_PATH)
        print(f"Data section identified: header at row {header_row_idx}, "
              f"total {num_block_rows} rows in block (incl. header).")

        tasks_dataframe, unique_tags = load_and_preprocess_tasks(
            DEFAULT_CSV_PATH,
            header_row_idx,
            num_block_rows
        )
        print(f"Tasks loaded and preprocessed. Found {len(unique_tags)} unique primary tags: {unique_tags}")
        if tasks_dataframe.empty and num_block_rows > 1 :
             print("Warning: Task data is empty after loading (header found but no data rows).")
        elif tasks_dataframe.empty:
             print("Warning: Task data is empty (no header or data rows).")

        chart_df = filter_aggregate_and_prepare_chart_data(
            tasks_dataframe,
            RELEVANT_DATE_START,
            unique_tags
        )
        print(f"Data filtered and aggregated for chart. {len(chart_df)} weeks of data.")

        create_and_save_bar_chart(
            chart_df,
            date_column_name="dateCompleted_week",
            output_filepath=OUTPUT_CHART_PATH
        )

    except FileNotFoundError as e:
        print(f"ERROR: File not found. {e}")
    except ValueError as e:
        print(f"ERROR: Invalid data or configuration. {e}")
    except Exception as e:
        print(f"ERROR: An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
