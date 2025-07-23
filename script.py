#%%
import polars as pl
import pulp
import time

start_time = time.time()

#%% Read and prepare data
print("Loading data...")
people = pl.read_csv("data/people.csv")

# People not going - add names here
not_going = [
    "Eleanor Clarke",
    "Anthony Bewes"
]

# Filter out people who are not going
going_people = people.filter(~pl.col('Name').is_in(not_going))
not_going_count = len(people) - len(going_people)

if not_going_count > 0:
    print(f"Excluding {not_going_count} people: {', '.join(not_going)}")
print(f"People participating: {len(going_people)}")

campers = going_people.select('Name').to_series().to_list()

cars_df = pl.scan_csv('data/cars.csv').with_columns(
    (pl.col('3-point belts (excluding driver)') + 1).alias('seats')
).collect()
cars = cars_df.select('Surname & Car Reg').to_series().to_list()

# Get all unique drivers from all driver columns, but only those who are going
all_drivers = []
for i in range(1, 7):  # Assuming up to 6 driver columns
    driver_col = f'Driver {i}'
    if driver_col in cars_df.columns:
        drivers_from_col = cars_df.select(driver_col).to_series().drop_nulls().to_list()
        all_drivers.extend(drivers_from_col)

# Only include drivers who are going (exist in going_people)
going_people_names = going_people.select('Name').to_series().to_list()
drivers = [d for d in set(all_drivers) if d in going_people_names]

print(f"Total potential drivers: {len(set(all_drivers))}")
print(f"Available drivers (going): {len(drivers)}")

# Create lookup dictionaries for performance (using only going people)
camper_to_group = dict(zip(going_people.select('Name').to_series(), going_people.select('Dorm').to_series()))
camper_to_gender = dict(zip(going_people.select('Name').to_series(), going_people.select('Sex').to_series()))
car_to_capacity = dict(zip(cars_df.select('Surname & Car Reg').to_series(), 
                          (cars_df.select('seats').to_series() - 1)))

# Pre-build driver-to-allowed-cars mapping
driver_allowed_cars = {}
for d in drivers:
    driver_allowed_cars[d] = []
    for row in cars_df.iter_rows(named=True):
        allowed_drivers = [row.get(f'Driver {x}') for x in range(1, 7) if row.get(f'Driver {x}') is not None]
        if d in allowed_drivers:
            driver_allowed_cars[d].append(row['Surname & Car Reg'])

# Get groups, leaders, minibus info (using only going people)
groups = going_people.select('Dorm').unique().to_series().to_list()
leaders = going_people.filter(pl.col('Year') == "L").select('Name').to_series().to_list()
minibus_cars = cars_df.filter(pl.col('Surname & Car Reg').str.contains('Minibus')).select('Surname & Car Reg').to_series().to_list()
male_campers = going_people.filter(pl.col('Sex') == 'M').select('Name').to_series().to_list()

print(f"Data loaded: {len(campers)} campers, {len(cars)} cars, {len(drivers)} drivers")

#%% Create optimization problem
print("Creating optimization problem...")
prob = pulp.LpProblem("Transport_Allocation", pulp.LpMaximize)

#%% Create decision variables
print("Creating decision variables...")

# Camper assigned to car
assign_camper = {}
for c in campers:
    for car in cars:
        var_name = f"camper_{c}_{car}".replace(' ', '_').replace('(', '').replace(')', '').replace('&', '')
        assign_camper[(c, car)] = pulp.LpVariable(var_name, cat='Binary')

# Driver assigned to car
assign_driver = {}
for d in drivers:
    for car in cars:
        var_name = f"driver_{d}_{car}".replace(' ', '_').replace('(', '').replace(')', '').replace('&', '')
        assign_driver[(d, car)] = pulp.LpVariable(var_name, cat='Binary')

#%% Set up objective function
print("Setting up objective function...")

# Base scores for all assignments
base_score = 50
prob += pulp.lpSum([base_score * assign_camper[(c, car)] for c in campers for car in cars])

# Group homogeneity scoring - reward based on minimum group size in each car
print("Adding group homogeneity bonuses...")

for car in cars:
    # Much simpler approach: just give bonuses based on group sizes directly
    # This avoids all variable multiplication issues
    
    capacity = car_to_capacity[car]
    car_groups_sizes = []
    
    for group in groups:
        group_members = going_people.filter(pl.col('Dorm') == group).select('Name').to_series().to_list()
        group_size_in_car = pulp.lpSum([assign_camper[(member, car)] for member in group_members])
        car_groups_sizes.append(group_size_in_car)
        
        # Give quadratic bonus for each group size
        # This naturally rewards: 2+2 over 3+1, and 4 over everything
        # 1 person: 1 point, 2 people: 4 points, 3 people: 9 points, 4 people: 16 points
        
        # Create binary variables for different group sizes to approximate quadratic scoring
        for size in range(1, capacity + 1):
            group_has_size = pulp.LpVariable(f"group_{group}_{car}_has_{size}".replace(' ', '_').replace('(', '').replace(')', '').replace('&', ''), 
                                           cat='Binary')
            
            # If group has exactly this size, give size^2 points
            # Link: group_has_size = 1 if group_size_in_car >= size
            prob += group_has_size <= group_size_in_car / size  # If size < size, then group_has_size = 0
            
            # Special scoring to prefer 2+2 over 3+1
            # Size 1: 5 points, Size 2: 50 points, Size 3: 60 points, Size 4: 200 points
            if size == 1:
                prob += group_has_size * 5
            elif size == 2:
                prob += group_has_size * 50  
            elif size == 3:
                prob += group_has_size * 60
            elif size == 4:
                prob += group_has_size * 200
    
    # Additional bonus for having fewer different groups (encourages consolidation)
    total_people_in_car = pulp.lpSum([assign_camper[(c, car)] for c in campers])
    
    # If car is used, give bonus for group consolidation
    car_is_used = pulp.LpVariable(f"car_{car}_used".replace(' ', '_').replace('(', '').replace(')', '').replace('&', ''), 
                                 cat='Binary')
    prob += car_is_used <= total_people_in_car  # If people > 0, car_is_used can be 1
    prob += total_people_in_car <= capacity * car_is_used  # If car_is_used = 0, no people
    
    # Bonus for using car (encourages filling cars rather than spreading thin)
    prob += car_is_used * 100

# Leader preference - using auxiliary variables
print("Adding leader preferences...")
leader_with_group = {}

for leader in leaders:
    leader_group = camper_to_group[leader]
    group_members = people.filter(pl.col('Dorm') == leader_group).select('Name').to_series().to_list()
    
    for car in cars:
        for member in group_members:
            if member != leader:
                var_name = f"leader_with_{leader}_{member}_{car}".replace(' ', '_').replace('(', '').replace(')', '').replace('&', '')
                leader_with_group[(leader, member, car)] = pulp.LpVariable(var_name, cat='Binary')
                
                # Linking constraints
                prob += leader_with_group[(leader, member, car)] <= assign_camper[(leader, car)]
                prob += leader_with_group[(leader, member, car)] <= assign_camper[(member, car)]
                prob += leader_with_group[(leader, member, car)] >= (
                    assign_camper[(leader, car)] + assign_camper[(member, car)] - 1
                )

# Add leader preference bonus to objective
prob += pulp.lpSum([15 * leader_with_group[(leader, member, car)] 
                   for (leader, member, car) in leader_with_group.keys()])

#%% Add constraints
print("Adding constraints...")

# Each camper in exactly one car (unless they're driving)
print("  - Camper assignment constraints...")
for c in campers:
    if c in drivers:
        # If person can drive, they're either passenger in one car OR driver of one car
        is_passenger = pulp.lpSum([assign_camper[(c, car)] for car in cars])
        is_driving = pulp.lpSum([assign_driver[(c, car)] for car in cars])
        prob += is_passenger + is_driving == 1
    else:
        # Regular campers must be passengers in exactly one car
        prob += pulp.lpSum([assign_camper[(c, car)] for car in cars]) == 1

# Each used car has exactly one driver (only if car has passengers)
print("  - Driver assignment constraints...")
for car in cars:
    car_has_passengers = pulp.lpSum([assign_camper[(c, car)] for c in campers])
    car_has_driver = pulp.lpSum([assign_driver[(d, car)] for d in drivers])
    
    # If car has passengers, it must have exactly one driver
    prob += car_has_driver >= car_has_passengers / len(campers)  # Forces driver if any passengers
    prob += car_has_driver <= car_has_passengers  # No driver if no passengers

# Each driver can only drive one car
print("  - Driver uniqueness constraints...")
for d in drivers:
    prob += pulp.lpSum([assign_driver[(d, car)] for car in cars]) <= 1

# People cannot be both driver and passenger (if they're in both lists)
print("  - Driver/passenger exclusion constraints...")
for person in drivers:
    if person in campers:  # Person is in both drivers and campers lists
        # If they're driving any car, they cannot be a passenger in any car
        is_driving = pulp.lpSum([assign_driver[(person, car)] for car in cars])
        is_passenger = pulp.lpSum([assign_camper[(person, car)] for car in cars])
        prob += is_driving + is_passenger <= 1

# Car capacity constraints
print("  - Capacity constraints...")
for car in cars:
    capacity = car_to_capacity[car]
    prob += pulp.lpSum([assign_camper[(c, car)] for c in campers]) <= capacity

# Driver can only drive authorized cars
print("  - Driver authorization constraints...")
for d in drivers:
    allowed_cars = driver_allowed_cars.get(d, [])
    for car in cars:
        if car not in allowed_cars:
            prob += assign_driver[(d, car)] == 0

# No male driver with all female passengers
print("  - Gender mixing constraints...")
for d in drivers:
    driver_gender = camper_to_gender.get(d, 'Unknown')
    if driver_gender == 'M':
        for car in cars:
            # If male driver assigned, at least one male camper must be in car
            prob += assign_driver[(d, car)] <= pulp.lpSum([assign_camper[(mc, car)] for mc in male_campers if mc in campers])

# Minibus needs at least 2 leaders
print("  - Minibus leader constraints...")
for car in minibus_cars:
    prob += pulp.lpSum([assign_camper[(leader, car)] for leader in leaders]) >= 2

#%% Solve the problem
print(f"Problem setup complete in {time.time() - start_time:.1f} seconds")
print("Starting optimization...")

solve_start = time.time()
# Try different solvers in order of preference
solvers_to_try = [
    ('PULP_CBC_CMD', lambda: prob.solve(pulp.PULP_CBC_CMD(msg=1, timeLimit=300))),
    ('Default', lambda: prob.solve()),
    ('GLPK', lambda: prob.solve(pulp.GLPK_CMD(msg=1))),
    ('COIN_CMD', lambda: prob.solve(pulp.COIN_CMD(msg=1, timeLimit=300)))
]

solved = False
for solver_name, solve_func in solvers_to_try:
    try:
        print(f"Trying {solver_name} solver...")
        solve_func()
        solved = True
        print(f"✅ Successfully solved with {solver_name}")
        break
    except Exception as e:
        print(f"❌ {solver_name} failed: {str(e)}")
        continue

if not solved:
    print("❌ All solvers failed!")
    exit(1)

solve_time = time.time() - solve_start
print(f"Optimization completed in {solve_time:.1f} seconds")

# Check solution status
print("Status:", pulp.LpStatus[prob.status])

if prob.status == pulp.LpStatusOptimal:
    print("✅ Optimal solution found!")
    print(f"Total score: {pulp.value(prob.objective):.0f}")
elif prob.status == pulp.LpStatusNotSolved:
    print("⏱️ Time limit reached - may have partial solution")
else:
    print("❌ No optimal solution found")
    print("Status code:", prob.status)

#%% Extract results
print("\nExtracting results...")

camper_assignments = {}
car_assignments = {}

for c in campers:
    for car in cars:
        if assign_camper[(c, car)].varValue == 1:
            camper_assignments[c] = car
            if car not in car_assignments:
                car_assignments[car] = []
            car_assignments[car].append(c)

driver_assignments = {}
for d in drivers:
    for car in cars:
        if assign_driver[(d, car)].varValue == 1:
            driver_assignments[car] = d

#%% Display results
print("\n" + "="*60)
print("TRANSPORT ALLOCATION RESULTS")
print("="*60)

total_passengers = 0
for car in sorted(cars):
    if car in car_assignments and car_assignments[car]:
        passengers = car_assignments[car]
        total_passengers += len(passengers)
        
        print(f"\n🚗 {car}")
        print(f"   Driver: {driver_assignments.get(car, '❌ NO DRIVER ASSIGNED')}")
        print(f"   Passengers ({len(passengers)}):")
        
        # Group passengers by dorm for readability
        passengers_by_group = {}
        for passenger in passengers:
            group = camper_to_group.get(passenger, 'Unknown')
            if group not in passengers_by_group:
                passengers_by_group[group] = []
            passengers_by_group[group].append(passenger)
        
        for group, members in sorted(passengers_by_group.items()):
            leader_count = sum(1 for m in members if m in leaders)
            leader_indicator = f" ({leader_count}L)" if leader_count > 0 else ""
            print(f"     {group}{leader_indicator}: {', '.join(sorted(members))}")

print(f"\nTotal passengers assigned: {total_passengers}/{len(campers)}")

#%% Validation
print("\n" + "="*60)
print("VALIDATION CHECKS")
print("="*60)

all_good = True

# Check capacity constraints
print("\n📊 CAPACITY CHECK:")
cars_used = 0
for car in sorted(cars):
    if car in car_assignments and car_assignments[car]:
        cars_used += 1
        capacity = car_to_capacity[car]
        actual = len(car_assignments[car])
        status = "✅" if actual <= capacity else "❌ OVER CAPACITY!"
        print(f"  {car}: {actual}/{capacity} passengers {status}")
        if actual > capacity:
            all_good = False

print(f"\nCars used: {cars_used}/{len(cars)} available")
unused_cars = [car for car in cars if car not in car_assignments or not car_assignments[car]]
if unused_cars:
    print(f"Unused cars: {', '.join(unused_cars[:5])}" + (f" ... and {len(unused_cars)-5} more" if len(unused_cars) > 5 else ""))

# Check all campers assigned
unassigned = [c for c in campers if c not in camper_assignments]
if unassigned:
    print(f"\n❌ UNASSIGNED CAMPERS ({len(unassigned)}):")
    for camper in unassigned:
        print(f"  - {camper}")
    all_good = False
else:
    print("\n✅ ALL CAMPERS ASSIGNED")

# Check group cohesion
print("\n👥 GROUP COHESION:")
for group in sorted(groups):
    group_members = people.filter(pl.col('Dorm') == group).select('Name').to_series().to_list()
    cars_used = set()
    for member in group_members:
        if member in camper_assignments:
            cars_used.add(camper_assignments[member])
    
    status = "✅" if len(cars_used) <= 2 else "⚠️"
    print(f"  {group}: {len(group_members)} people across {len(cars_used)} cars {status}")
    if len(cars_used) > 2:
        car_breakdown = {}
        for member in group_members:
            if member in camper_assignments:
                car = camper_assignments[member]
                if car not in car_breakdown:
                    car_breakdown[car] = []
                car_breakdown[car].append(member)
        for car, members in car_breakdown.items():
            print(f"    {car}: {', '.join(members)}")

# Check minibus leader requirements
print("\n👨‍💼 MINIBUS LEADER CHECK:")
for car in minibus_cars:
    if car in car_assignments:
        leaders_in_car = [p for p in car_assignments[car] if p in leaders]
        status = "✅" if len(leaders_in_car) >= 2 else "❌ NEEDS MORE LEADERS!"
        print(f"  {car}: {len(leaders_in_car)} leaders {status}")
        if leaders_in_car:
            print(f"    Leaders: {', '.join(leaders_in_car)}")
        if len(leaders_in_car) < 2:
            all_good = False

# Check driver assignments and driver/passenger conflicts
print("\n🚗 DRIVER ASSIGNMENT CHECK:")
driver_car_count = {}
driver_passenger_conflicts = []

for car, driver in driver_assignments.items():
    if driver not in driver_car_count:
        driver_car_count[driver] = []
    driver_car_count[driver].append(car)
    
    # Check if driver is also a passenger somewhere
    if driver in camper_assignments:
        passenger_car = camper_assignments[driver]
        driver_passenger_conflicts.append(f"  ❌ {driver} is driving {car} AND passenger in {passenger_car}")
        all_good = False

for driver, cars_driven in driver_car_count.items():
    if len(cars_driven) > 1:
        print(f"  ❌ {driver} assigned to {len(cars_driven)} cars: {', '.join(cars_driven)}")
        all_good = False
    else:
        print(f"  ✅ {driver}: {cars_driven[0]}")

if driver_passenger_conflicts:
    print("\n❌ DRIVER/PASSENGER CONFLICTS:")
    for conflict in driver_passenger_conflicts:
        print(conflict)
else:
    print("  ✅ No driver/passenger conflicts")

# Check for cars with passengers but no drivers (this would be a problem)
cars_needing_drivers = [car for car in car_assignments if car_assignments[car] and car not in driver_assignments]
if cars_needing_drivers:
    print(f"\n❌ CARS WITH PASSENGERS BUT NO DRIVER:")
    for car in cars_needing_drivers:
        print(f"  - {car}: {len(car_assignments[car])} passengers")
    all_good = False
else:
    print("✅ All cars with passengers have drivers")
print("\n🚻 GENDER MIXING CHECK:")
gender_issues = []
for car in car_assignments:
    driver = driver_assignments.get(car)
    if driver and camper_to_gender.get(driver) == 'M':
        passengers = car_assignments[car]
        male_passengers = [p for p in passengers if camper_to_gender.get(p) == 'M']
        female_passengers = [p for p in passengers if camper_to_gender.get(p) == 'F']
        
        if female_passengers and not male_passengers:
            gender_issues.append(f"  ❌ {car}: Male driver ({driver}) with only female passengers")
            all_good = False

if gender_issues:
    for issue in gender_issues:
        print(issue)
else:
    print("  ✅ No gender mixing issues found")

# Summary
print("\n" + "="*60)
if all_good:
    print("🎉 ALL VALIDATION CHECKS PASSED!")
else:
    print("⚠️  Some validation issues found - review above")

print(f"\nTotal runtime: {time.time() - start_time:.1f} seconds")
print("="*60)

#%% Export results (optional)
# Uncomment to save results
"""
import pandas as pd
results = []
for car in car_assignments:
    for passenger in car_assignments[car]:
        results.append({
            'car': car,
            'driver': driver_assignments.get(car, ''),
            'passenger': passenger,
            'group': camper_to_group.get(passenger, ''),
            'is_leader': passenger in leaders,
            'gender': camper_to_gender.get(passenger, '')
        })

results_df = pd.DataFrame(results)
results_df.to_csv('transport_allocation_results.csv', index=False)
print("Results exported to transport_allocation_results.csv")
"""