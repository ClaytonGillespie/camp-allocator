#%%
#!/usr/bin/env python3
"""
Manual Allocation Scorer
Scores a manual vehicle allocation using the same scoring system as the optimizer
"""

import pandas as pd
from collections import Counter

# Scoring configuration constants (same as in optimizer)
SCORE_WEIGHTS = {1: 1, 2: 8, 3: 27, 4: 64, 5: 125, 6: 216, 7: 250, 8: 250, 9: 250, 10: 250}
SINGLE_DORM_BONUS = 20  # Bonus for single-dorm cars (not minibuses)
GENDER_COHESION_BONUS = 10  # Bonus for single-gender mixed-dorm vehicles
MIN_SIZE_2_PENALTY = 500  # Penalty for min group size of 2
MIN_SIZE_1_PENALTY = 2000  # Penalty for min group size of 1


def score_manual_allocation(allocation_df, people_df, vehicles_df=None):
    """
    Score a manual allocation using the same scoring system as the optimizer.
    
    Args:
        allocation_df: DataFrame with vehicle names as index, 'Driver' column, 
                      and columns 0-16 containing passenger names (NaN for empty seats)
                      Example:
                          Index: 'Minibus (BG59 FDJ)'
                          Columns: Driver='Martin Loy', 0='Martin Loy', 1='Person 2', ..., 16=NaN
        people_df: DataFrame with columns ['Name', 'Dorm', 'Gender', 'Is Leader']
        vehicles_df: DataFrame with vehicle information (optional)
    
    Returns:
        Dictionary with detailed scoring breakdown
    """
    # Create lookup dictionaries
    people_to_dorm = dict(zip(people_df['Name'], people_df['Dorm']))
    people_to_gender = {}
    if 'Gender' in people_df.columns:
        people_to_gender = dict(zip(people_df['Name'], people_df['Gender']))
    
    # Results storage
    total_score = 0
    vehicle_scores = {}
    detailed_breakdown = []
    
    # Metrics
    total_people_allocated = 0
    vehicles_with_size_1 = 0
    vehicles_with_size_2 = 0
    single_dorm_cars = 0
    single_gender_mixed = 0
    isolated_people = 0
    
    # Process each vehicle (row in the dataframe)
    for vehicle in allocation_df.index:
        row = allocation_df.loc[vehicle]
        driver = row['Driver']
        
        # Collect passengers from columns 0-16, excluding NaN values
        # The driver is already in the 'Driver' column, so we include them separately
        passengers = [driver]  # Start with the driver
        
        for col in range(17):  # 0 to 16
            if str(col) in row.index and pd.notna(row[str(col)]):
                passengers.append(row[str(col)])
        
        if len(passengers) <= 1:  # Only driver, no other passengers
            continue
            
        # Count people by dorm
        dorm_counts = Counter(people_to_dorm.get(p, 'Unknown') for p in passengers)
        
        # Calculate minimum group size
        min_group_size = min(dorm_counts.values()) if dorm_counts else 0
        
        # Base score from minimum group size
        vehicle_score = SCORE_WEIGHTS.get(min_group_size, min_group_size * min_group_size)
        
        # Track penalties
        penalties = []
        bonuses = []
        
        # Apply penalties for small group sizes
        if min_group_size == 1:
            vehicle_score -= MIN_SIZE_1_PENALTY
            vehicles_with_size_1 += 1
            penalties.append(f"Min size 1: -{MIN_SIZE_1_PENALTY}")
            isolated_people += sum(1 for count in dorm_counts.values() if count == 1)
        elif min_group_size == 2:
            vehicle_score -= MIN_SIZE_2_PENALTY
            vehicles_with_size_2 += 1
            penalties.append(f"Min size 2: -{MIN_SIZE_2_PENALTY}")
            isolated_people += sum(1 for count in dorm_counts.values() if count == 1)
        else:
            isolated_people += sum(1 for count in dorm_counts.values() if count == 1)
        
        # Single dorm bonus (cars only)
        is_single_dorm = len(dorm_counts) == 1
        if is_single_dorm and 'Minibus' not in vehicle:
            vehicle_score += SINGLE_DORM_BONUS
            single_dorm_cars += 1
            bonuses.append(f"Single dorm car: +{SINGLE_DORM_BONUS}")
        
        # Gender cohesion bonus
        if people_to_gender and len(dorm_counts) > 1:
            gender_counts = Counter(people_to_gender.get(p, 'Unknown') for p in passengers)
            if len(gender_counts) == 1 and 'Unknown' not in gender_counts:
                vehicle_score += GENDER_COHESION_BONUS
                single_gender_mixed += 1
                bonuses.append(f"Gender cohesion: +{GENDER_COHESION_BONUS}")
        
        # Count leaders
        leaders = [p for p in passengers if people_df[people_df['Name'] == p]['Is Leader'].values[0]]
        
        # Store detailed info
        vehicle_info = {
            'vehicle': vehicle,
            'driver': driver,
            'passengers': len(passengers),
            'leaders': len(leaders),
            'min_group_size': min_group_size,
            'base_score': SCORE_WEIGHTS.get(min_group_size, min_group_size * min_group_size),
            'bonuses': bonuses,
            'penalties': penalties,
            'final_score': vehicle_score,
            'dorm_breakdown': dict(dorm_counts),
            'is_single_dorm': is_single_dorm,
            'is_minibus': 'Minibus' in vehicle
        }
        
        # Check minibus leader requirement
        if 'Minibus' in vehicle and len(leaders) < 2:
            vehicle_info['warning'] = f"Minibus has only {len(leaders)} leader(s), needs 2+"
        
        detailed_breakdown.append(vehicle_info)
        vehicle_scores[(vehicle, driver)] = vehicle_score
        total_score += vehicle_score
        total_people_allocated += len(passengers)
    
    # Summary
    summary = {
        'total_score': total_score,
        'vehicles_used': len(allocation_df),
        'people_allocated': total_people_allocated,
        'vehicles_with_size_1': vehicles_with_size_1,
        'vehicles_with_size_2': vehicles_with_size_2,
        'isolated_people': isolated_people,
        'single_dorm_cars': single_dorm_cars,
        'single_gender_mixed_dorms': single_gender_mixed,
        'average_score_per_vehicle': total_score / len(allocation_df) if len(allocation_df) > 0 else 0
    }
    
    return {
        'summary': summary,
        'vehicle_scores': vehicle_scores,
        'detailed_breakdown': detailed_breakdown
    }


def print_scoring_report(scoring_result):
    """Print a nicely formatted scoring report."""
    summary = scoring_result['summary']
    breakdown = scoring_result['detailed_breakdown']
    
    print("\n" + "="*60)
    print("ALLOCATION SCORING REPORT")
    print("="*60)
    
    print("\nSCORING RULES:")
    print(f"  Min group sizes: 1→{SCORE_WEIGHTS[1]}pt, 2→{SCORE_WEIGHTS[2]}pts, 3→{SCORE_WEIGHTS[3]}pts, 4→{SCORE_WEIGHTS[4]}pts, 5→{SCORE_WEIGHTS[5]}pts, etc.")
    print(f"  Single-dorm bonus: {SINGLE_DORM_BONUS}pts per car (not minibuses)")
    print(f"  Gender cohesion bonus: {GENDER_COHESION_BONUS}pts for mixed-dorm vehicles with single gender")
    print(f"  Group size penalties: -{MIN_SIZE_2_PENALTY}pts for size 2, -{MIN_SIZE_1_PENALTY}pts for size 1")
    
    print("\nSUMMARY:")
    print(f"  Total Score: {summary['total_score']}")
    print(f"  Vehicles Used: {summary['vehicles_used']}")
    print(f"  People Allocated: {summary['people_allocated']}")
    print(f"  Average Score per Vehicle: {summary['average_score_per_vehicle']:.1f}")
    
    if summary['vehicles_with_size_1'] > 0:
        print(f"  ⚠️  Vehicles with isolated people (size 1): {summary['vehicles_with_size_1']}")
    if summary['vehicles_with_size_2'] > 0:
        print(f"  ⚠️  Vehicles with min size 2: {summary['vehicles_with_size_2']}")
    
    print(f"  Single-dorm cars: {summary['single_dorm_cars']}")
    print(f"  Single-gender mixed-dorm vehicles: {summary['single_gender_mixed_dorms']}")
    print(f"  Total isolated people: {summary['isolated_people']}")
    
    print("\n" + "-"*60)
    print("VEHICLE BREAKDOWN:")
    print("-"*60)
    
    # Sort by score (worst first to highlight problems)
    sorted_vehicles = sorted(breakdown, key=lambda x: x['final_score'])
    
    for v in sorted_vehicles:
        print(f"\n{v['vehicle']} (Driver: {v['driver']})")
        print(f"  Passengers: {v['passengers']}, Leaders: {v['leaders']}")
        print(f"  Min group size: {v['min_group_size']} → Base score: {v['base_score']}")
        
        if v['bonuses']:
            print(f"  Bonuses: {', '.join(v['bonuses'])}")
        if v['penalties']:
            print(f"  Penalties: {', '.join(v['penalties'])}")
        
        print(f"  Final score: {v['final_score']} points")
        
        # Dorm breakdown
        print(f"  Dorms: ", end="")
        dorm_strs = [f"{dorm}:{count}" for dorm, count in sorted(v['dorm_breakdown'].items(), key=lambda x: -x[1])]
        print(", ".join(dorm_strs))
        
        if 'warning' in v:
            print(f"  ⚠️  {v['warning']}")


def convert_optimizer_output_to_dataframe(allocation_dict, vehicles_df):
    """
    Convert optimizer output format to the DataFrame format expected by the scorer.
    
    Args:
        allocation_dict: Dictionary from optimizer {(vehicle, driver): [passengers]}
        vehicles_df: DataFrame with vehicle capacities
    
    Returns:
        DataFrame with vehicle as index, Driver column, and columns 0-16
    """
    rows = []
    
    for (vehicle, driver), passengers in allocation_dict.items():
        # Create row data
        row_data = {'Driver': driver}
        
        # Fill passenger columns (driver is not in passengers list for this format)
        # Remove driver from passengers list if present
        passenger_list = [p for p in passengers if p != driver]
        
        for i in range(17):
            if i < len(passenger_list):
                row_data[str(i)] = passenger_list[i]
            else:
                row_data[str(i)] = None
        
        rows.append((vehicle, row_data))
    
    # Create DataFrame
    df = pd.DataFrame.from_dict(dict(rows), orient='index')
    
    # Ensure all columns exist in the right order
    columns = ['Driver'] + [str(i) for i in range(17)]
    df = df[columns]
    
    return df

#%% Load your data
allocation_df = pd.read_csv('manual.csv', index_col=0)
people_df = pd.read_csv('./data/campers.csv')
vehicles_df = pd.read_csv('./data/cars.csv')

# Score the allocation
result = score_manual_allocation(allocation_df, people_df,vehicles_df)
print_scoring_report(result)

#%%
# people = pl.read_csv("data/people.csv").with_columns(pl.when(pl.col('Year')=="L").then(True).otherwise(False).alias('Is Leader'))
