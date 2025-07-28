#%%
import polars as pl

def score_allocation(car_assignments, driver_assignments, people_df, cars_df):
    """
    Score a transport allocation using the same criteria as the optimizer
    
    Args:
        car_assignments: dict {car: [list of passengers]}
        driver_assignments: dict {car: driver_name}
        people_df: polars DataFrame with people data
        cars_df: polars DataFrame with car data
    
    Returns:
        dict with score breakdown and violations
    """
    
    # Create lookup dictionaries
    camper_to_group = dict(zip(people_df.select('Name').to_series(), people_df.select('Dorm').to_series()))
    camper_to_gender = dict(zip(people_df.select('Name').to_series(), people_df.select('Sex').to_series()))
    car_to_capacity = dict(zip(cars_df.select('Surname & Car Reg').to_series(), 
                              (cars_df.select('seats').to_series() - 1)))
    
    groups = people_df.select('Dorm').unique().to_series().to_list()
    leaders = people_df.filter(pl.col('Year') == "L").select('Name').to_series().to_list()
    minibus_cars = cars_df.filter(pl.col('Surname & Car Reg').str.contains('Minibus')).select('Surname & Car Reg').to_series().to_list()
    
    score_breakdown = {
        'base_score': 0,
        'group_cohesion': 0,
        'leader_preferences': 0,
        'total_score': 0
    }
    
    violations = []
    
    # 1. BASE SCORE - 50 points per assigned person
    total_passengers = sum(len(passengers) for passengers in car_assignments.values())
    score_breakdown['base_score'] = total_passengers * 50
    
    # 2. GROUP COHESION SCORE - 10 points per person in groups together
    for group in groups:
        group_members = people_df.filter(pl.col('Dorm') == group).select('Name').to_series().to_list()
        
        for car, passengers in car_assignments.items():
            group_members_in_car = [p for p in passengers if p in group_members]
            if len(group_members_in_car) > 0:
                # Bonus for each person when they're with group members
                score_breakdown['group_cohesion'] += len(group_members_in_car) * 10
    
    # 3. LEADER PREFERENCES - 15 points for each leader with group member
    for leader in leaders:
        if leader not in driver_assignments.values():  # Only if leader is passenger
            leader_group = camper_to_group.get(leader)
            if leader_group:
                group_members = people_df.filter(pl.col('Dorm') == leader_group).select('Name').to_series().to_list()
                
                # Find which car the leader is in
                leader_car = None
                for car, passengers in car_assignments.items():
                    if leader in passengers:
                        leader_car = car
                        break
                
                if leader_car:
                    # Count group members in same car (excluding leader)
                    group_members_in_car = [p for p in car_assignments[leader_car] 
                                          if p in group_members and p != leader]
                    score_breakdown['leader_preferences'] += len(group_members_in_car) * 15
    
    # 4. CHECK VIOLATIONS
    
    # Capacity violations
    for car, passengers in car_assignments.items():
        capacity = car_to_capacity.get(car, 0)
        if len(passengers) > capacity:
            violations.append(f"OVER CAPACITY: {car} has {len(passengers)} passengers, capacity {capacity}")
    
    # Driver/passenger conflicts
    for car, driver in driver_assignments.items():
        if car in car_assignments and driver in car_assignments[car]:
            violations.append(f"DRIVER CONFLICT: {driver} is both driving and passenger in {car}")
    
    # Male driver with all female passengers
    for car, driver in driver_assignments.items():
        driver_gender = camper_to_gender.get(driver, 'Unknown')
        if driver_gender == 'M' and car in car_assignments:
            passengers = car_assignments[car]
            male_passengers = [p for p in passengers if camper_to_gender.get(p) == 'M']
            female_passengers = [p for p in passengers if camper_to_gender.get(p) == 'F']
            
            if female_passengers and not male_passengers:
                violations.append(f"GENDER MIXING: Male driver {driver} with only female passengers in {car}")
    
    # Minibus leader requirements
    for car in minibus_cars:
        if car in car_assignments:
            passengers = car_assignments[car]
            leaders_in_car = [p for p in passengers if p in leaders]
            if len(leaders_in_car) < 2:
                violations.append(f"MINIBUS LEADERS: {car} has only {len(leaders_in_car)} leaders, needs 2+")
    
    # Check all campers assigned
    all_campers = people_df.select('Name').to_series().to_list()
    assigned_campers = set()
    for passengers in car_assignments.values():
        assigned_campers.update(passengers)
    for driver in driver_assignments.values():
        assigned_campers.add(driver)
    
    unassigned = [c for c in all_campers if c not in assigned_campers]
    if unassigned:
        violations.append(f"UNASSIGNED: {len(unassigned)} people not assigned: {', '.join(unassigned[:5])}")
    
    # Calculate total score (subtract penalty for violations)
    violation_penalty = len(violations) * 100  # 100 point penalty per violation
    score_breakdown['total_score'] = (score_breakdown['base_score'] + 
                                     score_breakdown['group_cohesion'] + 
                                     score_breakdown['leader_preferences'] - 
                                     violation_penalty)
    
    return {
        'score_breakdown': score_breakdown,
        'violations': violations,
        'total_passengers': total_passengers,
        'cars_used': len([car for car, passengers in car_assignments.items() if passengers]),
        'violation_penalty': violation_penalty
    }

def print_score_report(result):
    """Print a nice formatted report of the scoring"""
    print("=" * 60)
    print("TRANSPORT ALLOCATION SCORING REPORT")
    print("=" * 60)
    
    print(f"\n📊 SCORE BREAKDOWN:")
    print(f"  Base assignment score: {result['score_breakdown']['base_score']:,} points")
    print(f"  Group cohesion bonus:  {result['score_breakdown']['group_cohesion']:,} points")
    print(f"  Leader preference bonus: {result['score_breakdown']['leader_preferences']:,} points")
    
    if result['violation_penalty'] > 0:
        print(f"  Violation penalty:     -{result['violation_penalty']:,} points")
    
    print(f"  ─" * 30)
    print(f"  TOTAL SCORE:          {result['score_breakdown']['total_score']:,} points")
    
    print(f"\n📋 SUMMARY:")
    print(f"  Passengers assigned: {result['total_passengers']}")
    print(f"  Cars used: {result['cars_used']}")
    print(f"  Violations: {len(result['violations'])}")
    
    if result['violations']:
        print(f"\n❌ VIOLATIONS FOUND:")
        for violation in result['violations']:
            print(f"  - {violation}")
    else:
        print(f"\n✅ NO VIOLATIONS - Perfect allocation!")
    
    print("=" * 60)

#%% Load your data
not_going = ['Eleanor Clarke','Anthony Bewes']
people = pl.read_csv("data/people.csv"
            ).filter(
                pl.col('Name').is_in(not_going).not_()
            )
cars_df = pl.scan_csv('data/cars.csv').with_columns(
    (pl.col('3-point belts (excluding driver)') + 1).alias('seats')
).collect()

car_assignments = {
    'Minibus (HX70 BXB)':['Tom Gardner'
,'Jack Shannon'
,'Oliver Badger'
,'Rupert Fletcher'
,'Sam Jones'
,'George Mawby'
,'Finn Thomas'
,'Ben Youlten'
,'Alice McCaughern'
,'Pip Davies'
,'Isabel Roberts'
,'Rosie Lloyd Davies'
,'Trisha Salian'
,'Tanishka Nongbet Salian'
,'Allegra Sturgeon'
,'Mariia Babiichuk'
],
'Minibus (BL10 EYY)':['Tom Bromiley'
,'Jola Akin-Olugbade'
,'Kiki Akin-Olugbade'
,'Alex Boyd'
,'William Reeve'
,'Johnny Farrar-Bell'
,'Charlie Krammer'
,'Wilf Edwards'
,'Josh Cadbury'
,'Joel Sellers'
,'Beks Pentland'
,'Emily Williams'
,'Tilly Edgar'
,'Luka Holliday'
,'Ruthie Lewis'
,'Rosie Macmillan'
]
,
'Minibus (BG59 FDJ)':['Tom Haley'
,'James Krammer'
,'Harry Bewes'
,'Tom Bowen'
,'Toby Bromiley'
,'Joshua McCann'
,'Jack Nelson'
,'Oscar Runham'
,'Yang-Yang Guo'
,'Tabby Hill'
,'Hannah Shrives'
,'Bethia Smith'
,'Amelie Buckler'
,'Anna Bradley'
,'Rosie Howard'
,'Laura Nicholson'
],
'Weekes (FY13 EZA)':['Thomas Griffiths'
,'Jonathan Miles'
,'Sam Bradley'
],
'Johnson (YA64 FV2)':['Jonah Anderson'
,'Frazer Ashton'
,'Samuel Holliday'
],
'Bewes (FG58 TCU)':['Lily Middleditch'
,'Alice Montgomery'
,'Lucy Pitman'
],
'Cornes (WOLF)':['Sarah Kiaer'
,'Zoe Sellers'
,'Talitha Smith'
],
'Loy (BF70 AZT)':['Joanna Anderson'
,'Anne Kiragu'
,'Grace Roberts'
],
'Montgomery (VV24 MDN)':['Ellie Pereira'
,'Amalia Welbourn'
,'Ellie Jones'
],
'Thomas (LC24 V22)':['Sophie Cornes'
,'Brianna Howell'
,'Pippa Sanders'
],
'Mullins (PX67 LSN)':['Jack Youlten'
,'Jessy Banzoulou'
,'Ed Bewes'
],
'Davies (FG60 OUM)':['Titus Waldock'
,'George Chetwood'
,'Jasper Cliffe'
,'Sam Edwards'
],
'Jenkins (CK10 GZF)':['Brandon Howell'
,'Harry Telfer'
,'Will Thompson'
,'Milan Zhou'
]
}

driver_assignments= {
    'Minibus (HX70 BXB)':'Will Clarke'
    ,'Minibus (BL10 EYY)':'Martin Loy'
    ,'Minibus (BG59 FDJ)':'Paul Montgomery'
    ,'Weekes (FY13 EZA)':'Robin Weekes'
    ,'Johnson (YA64 FV2)':'Leo Johnson'
    ,'Bewes (FG58 TCU)':'Charlotte Bewes'
    ,'Cornes (WOLF)':'Alice Cornes'
    ,'Loy (BF70 AZT)':'Wendy Loy'
    ,'Montgomery (VV24 MDN)':'Charlie Montgomery'
    ,'Thomas (LC24 V22)':'Paget Thomas'
    ,'Mullins (PX67 LSN)':'Nick Mullins'
    ,'Davies (FG60 OUM)':'John Davies'
    ,'Jenkins (CK10 GZF)':'Rob Jenkins'
}

#%%
import pandas as pd
ca_df = pd.DataFrame.from_dict(car_assignments,orient='index')

replace_df = pd.read_csv('data/people_v2.csv')[['Name','Real Name']]
replace_dict = {}
for row in replace_df.iterrows():
    replace_dict[row[1][1]] = row[1][0]

for col in ca_df.columns:
    ca_df[col] = ca_df[col].map(replace_dict)
ca_df

da_df = pd.DataFrame.from_dict(driver_assignments,orient='index',columns=['Driver'])

final = ca_df.join(da_df)
final = final[['Driver'] + final.columns[:-1].tolist()]
final

#%%

# Score the allocation
result = score_allocation(car_assignments, driver_assignments, people, cars_df)
print_score_report(result)

#%%
# people = pl.read_csv("data/people.csv").with_columns(pl.when(pl.col('Year')=="L").then(True).otherwise(False).alias('Is Leader'))
