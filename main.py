import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from snowflake.snowpark import Session
from typing import Tuple, List, Dict, Any
from datetime import datetime

# --- Snowflake session ---
def get_snowflake_session() -> Session:
    return Session.builder.configs({
        "account": st.secrets["snowflake_account"],
        "user": st.secrets["snowflake_user"],
        "password": st.secrets["snowflake_password"],
        "role": st.secrets["snowflake_role"],
        "warehouse": st.secrets["snowflake_warehouse"],
        "database": st.secrets["snowflake_database"],
        "schema": st.secrets["snowflake_schema"]
    }).create()

# --- Reusable AG-Grid setup function ---
def setup_aggrid(ag_grid_key:str, df: pd.DataFrame, edit_mode: bool = False, enable_selection: bool = False, pagination_size: int = 1) -> Dict[str, Any]:
    """
    Reusable function to configure AG-Grid with consistent settings
    
    Args:
        df: DataFrame to display
        edit_mode: Whether editing is enabled
        enable_selection: Whether row selection is enabled
        pagination_size: Number of rows per page
    
    Returns:
        Dictionary with grid_options and other grid settings
    """
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_pagination(paginationPageSize=pagination_size)
    
    # Configure columns - make SALES_ACCOUNT_ID read-only always
    for col in df.columns:
        if col in ('index'):
            gb.configure_column(col, header_name="", editable=False, pinned='left', cellStyle={'color': '#0000'})
        else:
            gb.configure_column(col, editable=edit_mode, resizable=True)
    
    if enable_selection:
        gb.configure_selection("multiple", use_checkbox=True)
    
    gb.configure_grid_options(domLayout='normal')
    
    custom_css = {
                        ".ag-row-even": {"background-color": "#f0f0f0"},
                        ".ag-row-odd": {"background-color": "#ffffff"},
                        ".ag-header-cell": {"background-color": "#175388",
                                            "color": "white"}
                    }
    
    return {
        'grid_options': gb.build(),
        'update_mode': GridUpdateMode.MODEL_CHANGED if edit_mode else GridUpdateMode.NO_UPDATE,
        'allow_unsafe_jscode': True,
        'height': 400,
        'theme': 'streamlit',
        'custom_css': custom_css, 
        'key': ag_grid_key
    }

# --- Load all data from Snowflake (cached) ---
@st.cache_data
def load_all_data() -> pd.DataFrame:
    session = get_snowflake_session()
    df = session.table("ENRICHMENT").to_pandas()
    session.close()
    df.reset_index(inplace=True)
    return df

# --- Initialize session state ---
def init_session_state():
    if "edit_mode" not in st.session_state:
        st.session_state.edit_mode = True
    if "inserted_rows" not in st.session_state:
        st.session_state.inserted_rows = []
    if "original_df" not in st.session_state:
        st.session_state.original_df = None
    if "deleted_indices" not in st.session_state:
        st.session_state.deleted_indices = []
    if "modified_data" not in st.session_state:
        st.session_state.modified_data = None

# --- Get active data (not marked for deletion) ---
def get_active_data() -> pd.DataFrame:
    if st.session_state.original_df is None:
        return pd.DataFrame()
    
    # Filter out deleted indices
    active_indices = [i for i in st.session_state.original_df.index 
                     if i not in st.session_state.deleted_indices]
    
    if not active_indices:
        return pd.DataFrame(columns=st.session_state.original_df.columns)

    st.session_state["undeleted_original_df"] = st.session_state.original_df.iloc[active_indices]
    return st.session_state["undeleted_original_df"]

# --- Compare dataframes to find changes ---
def find_updated_rows(original_active_df: pd.DataFrame, current_df: pd.DataFrame) -> List[Dict[str, Any]]:  
    """Compare original active data and current data to find updated rows"""
    updated_rows = []
    
    if original_active_df is None or len(original_active_df) != len(current_df):
        return []
    
    # Compare each row
    for idx in range(len(current_df)):
        current_row = current_df.iloc[idx].to_dict()
        original_row = original_active_df.iloc[idx].to_dict()
        
        # Check if any values changed (excluding SALES_ACCOUNT_ID)
        current_filtered = {k: v for k, v in current_row.items() if k != 'SALES_ACCOUNT_ID'}
        original_filtered = {k: v for k, v in original_row.items() if k != 'SALES_ACCOUNT_ID'}
        
        if current_filtered != original_filtered:
            updated_rows.append(current_row)
    
    return updated_rows

def compare_dataframes(original_df, df_to_compare, inserted_row_ids):
    # Set 'ID' as the index for both DataFrames
    to_compare_original = original_df.set_index('index')
    to_compare_active = df_to_compare.set_index('index')
    
    # Align and compare rows with matching IDs
    comparison_result = to_compare_original.compare(to_compare_active)
    
    # Reset index to include 'ID' in the result
    comparison_result.reset_index(inplace=True)
    
    return comparison_result

def compare_dataframes_v2(original_df, df_to_compare, id_column):
    # Set 'ID' as the index for both DataFrames
    to_compare_original = original_df.set_index('index')
    to_compare_active = df_to_compare.set_index('index')
    
    # Align and compare rows with matching IDs
    comparison_result = to_compare_original.compare(to_compare_active)
    
    # Reset index to include 'ID' in the result
    comparison_result.reset_index(inplace=True)

    comparison_result_v2 = to_compare_original[id_column].compare(to_compare_active[id_column])
    
    return comparison_result, comparison_result_v2


def generate_update_statement(table_name: str, non_id_columns: List[str], id_column: str = "id = ?") -> str:
    set_clause = ", ".join(f"{col} = ?" for col in non_id_columns)
    update_statement = f"UPDATE {table_name} SET {set_clause} WHERE {id_column} = ?;"
    return update_statement

def format_updates(x):
    str_x = str(x)
    if str_x in ["nan","None",""]:
        return None
    else:
        return str_x
    
# --- Streamlit UI ---
def main() -> None:
    st.title("AG-Grid with Snowflake Integration & Index-Based Soft Delete")
    
    # Initialize session state
    init_session_state()
    
    # Load original data once and cache it
    st.session_state.original_df = load_all_data()
    
    if st.button("🔄 Refresh Data from Snowflake"):
        load_all_data.clear()
        st.session_state.deleted_indices = []
        st.session_state.inserted_rows = []
        st.rerun()

    # Tabs for main data and deletion queue
    # tab1 = st.tabs(["📊 Active Data"]) # tab2 , "🗑️ Deletion"
    
    # with tab1:
    # Get active data (not marked for deletion)
    active_df = get_active_data()
    if len(st.session_state.inserted_rows) > 0:
        active_df = pd.concat([active_df,pd.DataFrame(st.session_state.inserted_rows)])
    def reset_changes(disabled=False):
        if st.button("🔄 Reset Changes", disabled):
            st.session_state.inserted_rows = []
            st.session_state.deleted_indices = []
            st.session_state.current_aggrid_version += 1
            st.success("Changes reset")
            st.rerun()

    if active_df.empty:
        st.warning("No active data to display")
        reset_changes()
        return
    
    # Edit mode toggle
    col1, col2 = st.columns([1, 4])
    with col1:
        edit_mode = st.toggle("Edit Mode", value=st.session_state.edit_mode, key="edit_toggle")
        st.session_state.edit_mode = edit_mode
    
    with col2:
        if edit_mode:
            st.info("🔓 Edit mode is ON")
        else:
            st.info("🔒 Edit mode is OFF - Data is read-only")
    
    # Configure AG-Grid using reusable function
    active_df.sort_values("index",inplace=True)
    if "current_aggrid_version" not in st.session_state:
        st.session_state.current_aggrid_version = 1
    grid_config = setup_aggrid("current", active_df, edit_mode=edit_mode, enable_selection=edit_mode)
    
    # Display the grid
    grid_response = AgGrid(
        active_df,
        gridOptions=grid_config['grid_options'],
        update_mode=grid_config['update_mode'],
        allow_unsafe_jscode=grid_config['allow_unsafe_jscode'],
        height=grid_config['height'],
        theme=grid_config['theme'],
        custom_css=grid_config['custom_css']
    )
    
    # st.success(len(grid_response["data"]))
    # Only show edit controls when edit mode is on
    if edit_mode:
        st.subheader("Edit Controls")

        col1, col2, col3, col4 = st.columns(4)
        
        inserted_row_ids = [x["index"] for x in st.session_state.inserted_rows]
        
        with col1:
            if st.button("➕ Add Row", disabled=not edit_mode):
                if len(st.session_state.inserted_rows) > 0:
                    highest_id = st.session_state.inserted_rows[-1]["index"]
                else:
                    highest_id = st.session_state["original_df"]["index"].max() 
                new_id = highest_id + 1
                new_row = {col: "" for col in active_df.columns}
                new_row["index"] = new_id
                st.session_state.inserted_rows.append(new_row)
                st.rerun()
        
        with col2:
            selected_rows = grid_response.get("selected_rows", [])
            # Apply the patch for None type
            selected_rows = [] if type(selected_rows) == type(None) else selected_rows
            if st.button("🗑️ Delete Selected", disabled=not edit_mode or len(selected_rows) == 0):
                original_indices = selected_rows["index"].to_list()
                inserted_deletions = list(set(original_indices).intersection(inserted_row_ids))
                original_deletions = list(set(original_indices).difference(inserted_row_ids))
                # Add to deletion list
                st.session_state.deleted_indices.extend(original_deletions)
                st.session_state.deleted_indices = list(set(st.session_state.deleted_indices))

                if len(inserted_deletions) > 0:
                    st.session_state.inserted_rows = [
                        x for x in st.session_state.inserted_rows 
                        if x["index"] not in inserted_deletions
                    ]
                
                st.success(f"Moved {len(original_indices)} rows to deletion queue")
                st.rerun()
        
        with col3:
            if st.button("💾 Save Changes", disabled=not edit_mode):
                session = get_snowflake_session()
                id_column = "SALES_ACCOUNT_ID"
                table_name = "ENRICHMENT"

                # skip_ids = []
                # if len(active_df) != len(pd.unique(active_df[id_column])):
                #     duplicate_ids = active_df[active_df.duplicated(id_column, keep=False)][id_column].unique().tolist()
                #     skip_ids.extend(duplicate_ids)
                #     st.error(f"Duplicate IDs found: {duplicate_ids}. These will be skipped from saving.")

                # --- Deletions ---
                # Construct SQL DELETE query for Snowflake
                deleted_ids = st.session_state.deleted_indices
                if deleted_ids:
                    deleted_id_column_values = st.session_state.original_df.loc[st.session_state.original_df["index"].isin(deleted_ids)][id_column]
                    delete_query = f"""
                    DELETE FROM {table_name}
                    WHERE {id_column} IN ({','.join([f"'{id}'" for id in deleted_id_column_values])});
                    """
                    # st.success(delete_query)
                    session.sql(delete_query).collect()
                    # Execute this query using your Snowflake connection
                    # Example: conn.cursor().execute(delete_query)
                            
                # --- Find Modified Items ---
                df_to_compare = pd.DataFrame(grid_response["data"])
                df_to_compare = df_to_compare.loc[~df_to_compare["index"].isin(inserted_row_ids)]

                # st.success(st.session_state.undeleted_original_df.columns.index)
                # st.success(type(df_to_compare))

                comparison, comparison_v2 = compare_dataframes_v2(
                    st.session_state.undeleted_original_df,
                    df_to_compare,
                    id_column
                )

                # Extract modified IDs
                modified_indices = comparison.reset_index()["index"].to_list()  
                edits_df = pd.DataFrame(grid_response["data"])
                     
                if len(modified_indices) > 0:
                    modified_id_items = comparison_v2.reset_index()["index"].to_list()
                    if len(modified_id_items) > 0:
                        modified_id_items_id_column_values = st.session_state.original_df.loc[st.session_state.original_df["index"].isin(modified_id_items)][["index",id_column]].applymap(lambda x: str(x))
                        modified_id_items_id_column_new_values = edits_df.loc[edits_df["index"].isin(modified_id_items)][["index",id_column]].applymap(lambda x: str(x))

                        modified_and_original_id = modified_id_items_id_column_new_values.set_index("index").join(modified_id_items_id_column_values.set_index("index"),how="inner",rsuffix="_r").set_index(id_column)

                        for pairing in modified_and_original_id.to_records():
                            session.sql(f"""UPDATE {table_name}
                                SET {id_column} = ?
                                WHERE {id_column} = ?""",params=list(pairing)).collect()
                        
                    
                    modified_ready_df = edits_df.loc[edits_df["index"].isin(modified_indices)].drop("index",axis=1).applymap(format_updates)
                    # Move the id column to the end
                    non_id_columns = [x for x in modified_ready_df.columns if x != id_column]
                    modified_ready_df = modified_ready_df[non_id_columns + [id_column]]
                    update_statement = generate_update_statement(table_name,non_id_columns,id_column)
                    
                    for _, update_values in modified_ready_df.iterrows():
                        session.sql(update_statement,params=list(update_values)).collect() 
                    
                inserted_df = edits_df.loc[edits_df["index"].isin(inserted_row_ids)].applymap(format_updates)
                inserted_df.drop("index",axis=1,inplace=True)
                insert_statement = f"""INSERT INTO {table_name} VALUES ({",".join(["?"] * len(inserted_df.columns))})"""
                for _, row in inserted_df.iterrows():
                    session.sql(insert_statement,row.to_list()).collect()

                session.close()
                load_all_data.clear()
                st.session_state.deleted_indices = []
                st.session_state.inserted_rows = []
                # st.rerun()

        with col4:
            reset_changes(not edit_mode)
    else:
        inserted_row_ids = []
    with st.expander("📊 Change Statistics"):
        st.write(f"**Deleted Rows:** {len(st.session_state.deleted_indices)}")
        # if st.session_state.deleted_indices:
        #     st.write(f"Rows Gett: {sorted(st.session_state.deleted_indices)}")
        st.write(f"**Inserted Rows:** {len(st.session_state.inserted_rows)}")
        # Modifications
        df_to_compare = pd.DataFrame(grid_response["data"])
        df_to_compare = df_to_compare.loc[~df_to_compare["index"].isin(inserted_row_ids)]
        modified_count = len(compare_dataframes(st.session_state.undeleted_original_df,df_to_compare,inserted_row_ids))
        st.write(f"**Modified Rows:** {modified_count}")
        
        

        # st.success(st.session_state.original_df active_df.loc[~active_df["ID"].isin(inserted_row_ids)])

    # id_column = "SALES_ACCOUNT_ID"



if __name__ == "__main__":
    main()
    
                


# session = get_snowflake_session()
# sql_query = """INSERT INTO ENRICHMENT VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""" 

# parameters = [None, 'Testing', None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
# session.sql(sql_query,params=parameters).collect()