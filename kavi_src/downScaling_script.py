import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import geopandas as gpd
from shapely.geometry import Point, polygon
import os
from shapely import wkt
import matplotlib.pyplot as plt

capacity_file = r"sample_data\combined_capacity_MGA_min.csv"#path to cluster specific capacity result for solar for each region file
CPAID_lcoe_pv = r"sample_data\solar_lcoe_ipm_metro_county.csv" #path to map of cluster to cpa connection file
CPAID_lcoe_onshore = r"sample_data\onshorewind_lcoe_ipm_metro_county.csv" #path to map of cluster to cpa connection file
folder_path = r"sample_data\extra_outputs\extra_outputs" 
CandidateProjectArea_SolarPV=pd.read_csv(r"sample_data\CandidateProjectArea_SolarPV.csv")
CandidateProjectArea_OnshoreWind=pd.read_csv(r"sample_data\CandidateProjectArea_OnshoreWind.csv")


def load_lcoe_data() -> pd.DataFrame:
    """Load capacity data from the given file."""
    lcoe_df_pv = pd.read_csv(CPAID_lcoe_pv, low_memory=False)
    lcoe_df_onshore = pd.read_csv(CPAID_lcoe_onshore, low_memory=False)
    return lcoe_df_pv, lcoe_df_onshore

def load_capacity_data(cluster_file: str) -> pd.DataFrame:
    """Load cluster assignments from the given file."""
    df = pd.read_csv(cluster_file)
    generator_data=pd.read_csv(r"C:\Users\kavi5\Enhancing_Resilient_Solar_Power\git\down_scaling\sample_data\Generators_data.csv")

    if 'iter' not in df.columns:
        iteration_number = 0
        in_iteration = False
        iteration_numbers = []
        for index, row in df.iterrows():
            if 'NENG_Rest_battery_moderate_0' in row.values:
                iteration_number += 1
                in_iteration = True
            elif 'Total' in row.values:
                in_iteration = False

            if in_iteration or ('Total' in row.values and not in_iteration):
                iteration_numbers.append(iteration_number)
            else:
                iteration_numbers.append(None)

        df['iter'] = iteration_numbers

    df = df[df['Resource'].str.contains('_landbasedwind|utilitypv', case=False, na=False)]
    
    condition_utilitypv = df['Resource'].str.contains('utilitypv', case=False, na=False)

    df.loc[condition_utilitypv, "Cluster"] = df.loc[condition_utilitypv, "Resource"].str.extract(r'moderate_(\d+)', expand=False)
    df.loc[~condition_utilitypv, "Cluster"] = df.loc[~condition_utilitypv, "Resource"].str.extract(r'landbasedwind_class1_moderate_(\d+)', expand=False)

    generator_data = generator_data.drop_duplicates(subset=['Zone'], keep='first')
    df = pd.merge(df, generator_data[['Zone', 'region']], on=['Zone'], how='left')
    df['technology'] = df['Resource'].apply(lambda x: 'UtilityPV_Class1_Moderate_' if 'utilitypv' in x.lower() else 'LandbasedWind')
    df.to_csv("capacity_alter.csv")
    return df


def randomly_select_cluster( df : pd.DataFrame, tech: str):
    
    pop_density_ranges ={ "utilitypv": [
        (0, 5, 0.2),
        (5, 10, 0.1),
        (10, 30, 0.05),
        (30, 40, 0.025),
        (40, 60, 0.02),
        (60, float('inf'), 0)
    ],
    "onshore_wind":[
         (0, 1, 0.6),
        (1, 3, 0.2),
        (3, 5, 0.1),
        (5, 10, 0.05),
        (10, float('inf'), 0)
    ]
    }
    cluster_factor = {"utilitypv": 1.1, "onshore_wind": 6}
    n_clusters = max(1, int(len(df) // cluster_factor[tech]))

    coords = df[['latitude', 'longitude']].values
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(coords)
    Kmeans =  MiniBatchKMeans(n_clusters=n_clusters, batch_size=10000, random_state=0, n_init='auto')
    Kmeans=Kmeans.fit(scaled_data)

    df['cluster_mean'] = Kmeans.labels_

    sampled_dfs = []
    for min_popden, max_popden, sample_frac in pop_density_ranges[tech]:

        df_subset = df[(df['m_popden'] >= min_popden) & (df['m_popden'] < max_popden)]
        selected_clust = (
            pd.DataFrame(df_subset["cluster_mean"].unique())
            .sample(frac=sample_frac, replace=False)
            .rename(columns={0: "cluster_mean"})
        )
        final_df = df_subset.loc[
            df_subset["cluster_mean"].isin(selected_clust["cluster_mean"])
        ]
        sampled_dfs.append(final_df)

    # Combine all the sampled and merged dataframes
    merged_df = pd.concat(sampled_dfs)
    merged_df.reset_index(drop=True, inplace=True)
    return merged_df

def find_and_read_matching_file(region, technology):
    
    # Construct the pattern to search for in file names
    pattern = f"{region}_{technology.split('_')[0]}"
    
    # Check all files in the folder
    matching_files = [file for file in os.listdir(folder_path) if pattern in file]
    
    if matching_files:
        file_name = matching_files[0] 
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_csv(file_path)
        return df
    else:
        return pd.DataFrame(), None  # Return an empty DataFrame and None if no matching file found


def select_lcoe_cpas(
    cluster: int,
    cluster_capacity: float,
    technology: str,
    region:str,
    lcoe_df_pv : pd.DataFrame, 
    lcoe_df_onshore: pd.DataFrame
) -> pd.DataFrame:
    """Select CPAs based on LCOE values and cluster capacity."""
    # Normalize column names to a consistent case (lowercase in this example)
    if(technology=="UtilityPV_Class1_Moderate_"):
        lcoe_df=lcoe_df_pv
    else:
        lcoe_df=lcoe_df_onshore
    cluster_assignments_df=find_and_read_matching_file(region,technology)
    cluster_cpas = cluster_assignments_df[cluster_assignments_df["cluster"] == cluster]["cpa_id"]
    cluster_lcoe_df = lcoe_df[lcoe_df["CPA_ID"].isin(cluster_cpas)]
    cluster_lcoe_df = cluster_lcoe_df.sort_values(by="lcoe")
    
    cumulative_capacity = 0
    selected_cpas = []
    for _, row in cluster_lcoe_df.iterrows():
        cumulative_capacity += row["cpa_mw"]
        selected_cpas.append(row)
        if cumulative_capacity >= cluster_capacity:
            break
    
    selected_cpas_df = pd.DataFrame(selected_cpas)
    return selected_cpas_df

def optimize_program(capacity_file: str) -> None:

    capacity_df = load_capacity_data(capacity_file)
    lcoe_df_pv, lcoe_df_onshore = load_lcoe_data()
    iter = capacity_df["iter"].unique()
    selected_cpas_1 = []
    selected_cpas_2 = []
    desired_columns = [
    'Cluster', 'region', 'technology', 'Resource', 'Zone', 'EndCap', 
    'EndEnergyCap', 'EndChargeCap', 'NewCap', 'RetCap', 'NewEnergyCap', 
    'RetEnergyCap', 'NewChargeCap', 'RetChargeCap', 'iter', 
     'CPA_ID', 'lcoe', 'Area', 'latitude', 'longitude', 
    'interconnect_annuity', 'state', 'm_landcover', 'm_primeFarmland', 
    'exFacil', 'plFacil', 'm_HMI', 'm_popden', 'cpa_mw']
    for iter_number in iter:
        zone_capacity_df = capacity_df[(capacity_df["iter"] == float(iter_number))].copy()
        iter_selected_cpas = []
        for intex, row in zone_capacity_df.iterrows():
            cluster_capacity = row["NewCap"]
            cluster=row['Cluster']
            technology=row['technology']
            region=row['region']
            selected_cpas_df = select_lcoe_cpas(int(cluster), cluster_capacity,technology,region, lcoe_df_pv, lcoe_df_onshore)
            selected_cpas_df["Zone"] = row['Zone']
            selected_cpas_df["Resource"] = row['Resource']
            selected_cpas_df["technology"] = row['technology']
            selected_cpas_df["iter"] = row['iter']
            iter_selected_cpas.append(selected_cpas_df)
            selected_cpas_2.append(selected_cpas_df)


        
       # Convert the list of dataframes to a single dataframe for this iteration
        iter_selected_cpas_df = pd.concat(iter_selected_cpas, ignore_index=True)
        # Filtering and random selection for UtilityPV_Class1_Moderate_
        selected_solar_cpas_df = iter_selected_cpas_df[iter_selected_cpas_df['technology'] == 'UtilityPV_Class1_Moderate_']
        if len(selected_solar_cpas_df.index) > 1:
            selected_solar_cpas_df =  randomly_select_cluster(selected_solar_cpas_df, 'utilitypv')
        selected_cpas_1.append(selected_solar_cpas_df)

        # Filtering and random selection for LandbasedWind
        selected_wind_cpas_df = iter_selected_cpas_df[iter_selected_cpas_df['technology'] == 'LandbasedWind']
        if len(selected_wind_cpas_df.index) > 1:
            selected_wind_cpas_df = randomly_select_cluster(selected_wind_cpas_df, 'onshore_wind')
        selected_cpas_1.append(selected_wind_cpas_df)

    # Convert the list of dataframes to a single dataframe after the loop if needed
    derated_selected_cpas_df = pd.concat(selected_cpas_1, ignore_index=True)
    derated_selected_cpas = pd.merge(derated_selected_cpas_df,   capacity_df, on=["Resource","Zone","iter","technology"], how='inner') 
    derated_selected_cpas = derated_selected_cpas[desired_columns]
    derated_selected_cpas.to_csv(r"output\Derated_selected_CPAs.csv", index=False)

    final_selected_cpas_df = pd.concat(selected_cpas_2, ignore_index=True)
    final_selected_cpas = pd.merge(final_selected_cpas_df,   capacity_df, on=["Resource","Zone","iter","technology"], how='inner') 
    final_selected_cpas = final_selected_cpas[desired_columns]
    final_selected_cpas.to_csv(r"output\All_selected_CPAs.csv", index=False)



    print(type(derated_selected_cpas))
    selected_solar = derated_selected_cpas[derated_selected_cpas['technology'] == 'UtilityPV_Class1_Moderate_']
    selected_solar=pd.merge(selected_solar, CandidateProjectArea_SolarPV[['CPA_ID', 'geometry']], on='CPA_ID', how="left")
    selected_solar.to_csv("selected_cpa_solar_geo.csv")
    selected_solar['geometry'] = selected_solar['geometry'].apply(wkt.loads)
    solar_gdf = gpd.GeoDataFrame(selected_solar, geometry='geometry')
    solar_gdf.set_crs(epsg=5070, inplace=True)  # WGS84

    output_folder = 'shapefile_solar'
    os.makedirs(output_folder, exist_ok=True)

    shapefile_path = os.path.join(output_folder, f'downScaled_CPAs.shp')
    solar_gdf.to_file(shapefile_path, driver='ESRI Shapefile')

    print(f"Shapefile saved to {shapefile_path}")

    
    selected_wind = derated_selected_cpas[derated_selected_cpas['technology'] == 'LandbasedWind']
    selected_wind=pd.merge(selected_wind, CandidateProjectArea_OnshoreWind[['CPA_ID', 'geometry']], on='CPA_ID', how="left")
    selected_wind.to_csv("selected_cpa_wind_geo.csv")
    
    selected_wind['geometry'] = selected_wind['geometry'].apply(wkt.loads)
    wind_gdf = gpd.GeoDataFrame(selected_wind, geometry='geometry')
    wind_gdf.set_crs(epsg=5070, inplace=True)  # WGS84

    output_folder = 'shapefile_wind'
    os.makedirs(output_folder, exist_ok=True)

    shapefile_path = os.path.join(output_folder, f'downScaled_CPAs.shp')
    wind_gdf.to_file(shapefile_path, driver='ESRI Shapefile')

    print(f"Shapefile saved to {shapefile_path}")

   # Load IPM regions
    IPM_regions = gpd.read_file(r"sample_data\IPM Regions v617 04-05-17\IPM_Regions_201770405.shp")
    target_crs = 'EPSG:4326'
    IPM_regions = IPM_regions.to_crs(target_crs)
    solar_gdf = solar_gdf.to_crs(target_crs)
    wind_gdf = wind_gdf.to_crs(target_crs)

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 12))
    IPM_regions.plot(ax=ax, color='#EBD5E7', edgecolor='#EBD5E7', alpha=0.5, label='IPM Regions')
    solar_gdf.plot(ax=ax, color='red', edgecolor='red', alpha=0.5, label='Solar')
    wind_gdf.plot(ax=ax, color='blue', edgecolor='blue', alpha=0.5, label='Wind')

    ax.set_title('Down Scaled Data')
    ax.legend(title='Legend', loc='upper right')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    plt.savefig('output_image_1.pdf')
    plt.show()




if __name__=="__main__": 
    optimize_program(capacity_file)


