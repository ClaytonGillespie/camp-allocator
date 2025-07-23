#%%
import polars as pl
import pulp
import time
import pandas as pd

#%% Read and prepare data
start_time = time.time()
print("Loading data...")
not_going = ['Eleanor Clarke','Anthony Bewes']
people = pl.read_csv("data/people.csv"
            ).filter(
                pl.col('Name').is_in(not_going).not_()
            )
campers = people.select('Name').to_series().to_list()

cars_df = pl.scan_csv('data/cars.csv').with_columns(
    (pl.col('3-point belts (excluding driver)') + 1).alias('seats')
).collect()
cars = cars_df.select('Surname & Car Reg').to_series().to_list()

start_time = time.time()

#%% Get all unique drivers from all driver columns
all_drivers = []
for i in range(1, 7):  # Assuming up to 6 driver columns
    driver_col = f'Driver {i}'
    if driver_col in cars_df.columns:
        drivers_from_col = cars_df.select(driver_col).to_series().drop_nulls().to_list()
        all_drivers.extend(drivers_from_col)
drivers = list(set(all_drivers))
drivers = [d for d in drivers if d not in not_going]

# Create lookup dictionaries for performance
camper_to_group = dict(zip(people.select('Name').to_series(), people.select('Dorm').to_series()))
camper_to_gender = dict(zip(people.select('Name').to_series(), people.select('Sex').to_series()))
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

# Get groups, leaders, minibus info
groups = people.select('Dorm').unique().to_series().to_list()
leaders = people.filter(pl.col('Year') == "L").select('Name').to_series().to_list()
minibus_cars = cars_df.filter(pl.col('Surname & Car Reg').str.contains('Minibus')).select('Surname & Car Reg').to_series().to_list()
male_campers = people.filter(pl.col('Sex') == 'M').select('Name').to_series().to_list()

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

# Simplified group cohesion - bonus for larger groups in same car
print("Adding group cohesion bonuses...")
for group in groups:
    group_members = people.filter(pl.col('Dorm') == group).select('Name').to_series().to_list()
    for car in cars:
        group_size_in_car = pulp.lpSum([assign_camper[(member, car)] for member in group_members])
        # Quadratic bonus for keeping groups together
        prob += group_size_in_car * 10

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

# Each camper in exactly one car
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

# NEW SECTION - add after driver uniqueness constraints:
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
    ('PULP_CBC_CMD', lambda: prob.solve(pulp.PULP_CBC_CMD(msg=1, timeLimit=300)))
    ,('Default', lambda: prob.solve())
    ,('GLPK', lambda: prob.solve(pulp.GLPK_CMD(msg=1)))
    ,('COIN_CMD', lambda: prob.solve(pulp.COIN_CMD(msg=1, timeLimit=300)))
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

#%%

ca_df = pd.DataFrame.from_dict(car_assignments,orient='index')
da_df = pd.DataFrame.from_dict(driver_assignments,orient='index',columns=['Driver'])

final = ca_df.join(da_df)
final = final[['Driver']+final.columns[:-1].tolist()]
# final.to_csv('latest_transport_allocation_results.csv', index=False)
print("Results exported to transport_allocation_results.csv")
final.to_clipboard()

# SCORE BREAKDOWN:
#   Base assignment score: 4,100 points
#   Group cohesion bonus:  820 points
#   Leader preference bonus: 1,740 points
#   ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─
#   TOTAL SCORE:          6,660 points

# 📋 SUMMARY:
#   Passengers assigned: 82
#   Cars used: 11
#   Violations: 0

# ✅ NO VIOLATIONS - Perfect allocation!
# ============================================================

#%% Validation ######################################
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
unassigned = [u for u in unassigned if u not in driver_assignments.values()]
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

# Check driver assignments
print("\n🚗 DRIVER ASSIGNMENT CHECK:")
driver_passenger_conflicts = []
driver_car_count = {}
for car, driver in driver_assignments.items():
    if driver not in driver_car_count:
        driver_car_count[driver] = []
    driver_car_count[driver].append(car)

    if driver in camper_assignments:
        passenger_car = camper_assignments[driver]
        driver_passenger_conflicts.append(f"  ❌ {driver} is driving {car} AND passenger in {passenger_car}")
        all_good = False
    
    if driver_passenger_conflicts:
        print("\n❌ DRIVER/PASSENGER CONFLICTS:")
        for conflict in driver_passenger_conflicts:
            print(conflict)
    else:
        print("  ✅ No driver/passenger conflicts")

for driver, cars_driven in driver_car_count.items():
    if len(cars_driven) > 1:
        print(f"  ❌ {driver} assigned to {len(cars_driven)} cars: {', '.join(cars_driven)}")
        all_good = False
    else:
        print(f"  ✅ {driver}: {cars_driven[0]}")

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

# %%
