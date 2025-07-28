#%%
import polars as pl
import pulp
import time

start_time = time.time()

#%% Read and prepare data
print("Loading data...")
people = pl.read_csv("data/people_anon.csv")

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

# All participants are potential passengers (including those who might drive)
all_participants = going_people.select('Name').to_series().to_list()

cars_df = pl.scan_csv('data/cars.csv').with_columns(
    (pl.col('3-point belts (excluding driver)') + 1).alias('seats')
).collect()
cars = cars_df.select(pl.col('Surname & Car Reg')).to_series().to_list()

# Get all unique drivers from all driver columns, but only those who are going
all_potential_drivers = []
for i in range(1, 7):  # Assuming up to 6 driver columns
    driver_col = f'Driver {i}'
    if driver_col in cars_df.columns:
        drivers_from_col = cars_df.select(driver_col).to_series().drop_nulls().to_list()
        all_potential_drivers.extend(drivers_from_col)

# Only include drivers who are going (exist in going_people)
going_people_names = going_people.select('Name').to_series().to_list()
available_drivers = [d for d in set(all_potential_drivers) if d in going_people_names]

print(f"Total potential drivers: {len(set(all_potential_drivers))}")
print(f"Available drivers (going): {len(available_drivers)}")
print(f"Total people to transport: {len(all_participants)}")

# Create lookup dictionaries for performance (using only going people)
participant_to_group = dict(zip(going_people.select('Name').to_series(), going_people.select('Dorm').to_series()))
participant_to_gender = dict(zip(going_people.select('Name').to_series(), going_people.select('Sex').to_series()))
car_to_capacity = dict(zip(cars_df.select('Surname & Car Reg').to_series(), 
                          (cars_df.select('seats').to_series() - 1)))

# Pre-build driver-to-allowed-cars mapping
driver_allowed_cars = {}
for d in available_drivers:
    driver_allowed_cars[d] = []
    for row in cars_df.iter_rows(named=True):
        allowed_drivers = [row.get(f'Driver {x}') for x in range(1, 7) if row.get(f'Driver {x}') is not None]
        if d in allowed_drivers:
            driver_allowed_cars[d].append(row['Surname & Car Reg'])

# Get groups, leaders, minibus info (using only going people)
groups = going_people.select('Dorm').unique().to_series().to_list()
leaders = going_people.filter(pl.col('Is Leader') == True).select('Name').to_series().to_list()
minibus_cars = cars_df.filter(pl.col('Surname & Car Reg').str.contains('Minibus')).select('Surname & Car Reg').to_series().to_list()
male_participants = going_people.filter(pl.col('Sex') == 'M').select('Name').to_series().to_list()

print(f"Groups: {len(groups)}, Leaders: {len(leaders)}, Minibuses: {len(minibus_cars)}")

#%% Create optimization problem
print("Creating optimization problem...")
prob = pulp.LpProblem("Transport_Allocation", pulp.LpMaximize)

#%% Create decision variables
print("Creating decision variables...")

# Each participant can be assigned as a passenger to a car
assign_passenger = {}
for person in all_participants:
    for car in cars:
        var_name = f"passenger_{person}_{car}".replace(' ', '_').replace('(', '').replace(')', '').replace('&', '')
        assign_passenger[(person, car)] = pulp.LpVariable(var_name, cat='Binary')

# Each available driver can be assigned to drive a car
assign_driver = {}
for driver in available_drivers:
    for car in cars:
        var_name = f"driver_{driver}_{car}".replace(' ', '_').replace('(', '').replace(')', '').replace('&', '')
        assign_driver[(driver, car)] = pulp.LpVariable(var_name, cat='Binary')

print(f"Created variables: {len(all_participants)} participants, {len(available_drivers)} drivers, {len(cars)} cars")

#%% Set up objective function - ULTRA SIMPLE VERSION
print("Setting up ultra-simple objective function...")

# Base scores for all assignments
base_score = 50
prob += pulp.lpSum([base_score * assign_passenger[(person, car)] for person in all_participants for car in cars])
prob += pulp.lpSum([base_score * assign_driver[(driver, car)] for driver in available_drivers for car in cars])

# Ultra-simple group bonuses - no auxiliary variables
print("Adding ultra-simple group bonuses...")
for car in cars:
    for group in groups:
        group_members = going_people.filter(pl.col('Dorm') == group).select('Name').to_series().to_list()
        
        # Count passengers from this group in this car
        passengers_from_group = pulp.lpSum([assign_passenger[(member, car)] for member in group_members])
        
        # Count if leader from this group is driving this car
        drivers_from_group = pulp.lpSum([assign_driver[(member, car)] for member in group_members if member in available_drivers])
        
        # Total people from this group in this car (passengers + driver)
        total_from_group = passengers_from_group + drivers_from_group
        
        # Simple exponential bonus for group size
        # 1 person: 1000 points
        # 2 people: 4000 points  
        # 3 people: 9000 points
        # 4 people: 16000 points
        # 5+ people: 25000+ points
        prob += total_from_group * * 1000

# Extra leader bonus - simple linear approach
print("Adding simple leader bonuses...")
for leader in leaders:
    if leader in available_drivers:
        leader_room = participant_to_group.get(leader)
        if leader_room:
            room_members = going_people.filter(pl.col('Dorm') == leader_room).select('Name').to_series().to_list()
            
            for car in cars:
                # If leader drives this car, bonus for each room member passenger
                leader_driving = assign_driver[(leader, car)]
                room_passengers = pulp.lpSum([assign_passenger[(member, car)] for member in room_members if member != leader])
                
                # Linear bonus: 10000 points per room member when leader drives
                # This works because we're multiplying a binary (leader_driving) by a sum
                prob += leader_driving * room_passengers * 10000

#%% Add constraints
print("Adding constraints...")

# Each participant must be either a passenger in one car OR a driver of one car (not both, not neither)
print("  - Participant assignment constraints...")
for person in all_participants:
    is_passenger = pulp.lpSum([assign_passenger[(person, car)] for car in cars])
    
    if person in available_drivers:
        # If person can drive, they're either passenger in one car OR driver of one car
        is_driver = pulp.lpSum([assign_driver[(person, car)] for car in cars])
        prob += is_passenger + is_driver == 1
    else:
        # If person cannot drive, they must be passenger in exactly one car
        prob += is_passenger == 1

# Each used car has exactly one driver (only if car has passengers)
print("  - Driver assignment constraints...")
for car in cars:
    car_has_passengers = pulp.lpSum([assign_passenger[(person, car)] for person in all_participants])
    car_has_driver = pulp.lpSum([assign_driver[(driver, car)] for driver in available_drivers])
    
    # If car has passengers, it must have exactly one driver
    prob += car_has_driver >= car_has_passengers / len(all_participants)  # Forces driver if any passengers
    prob += car_has_driver <= car_has_passengers  # No driver if no passengers

# Each driver can only drive one car
print("  - Driver uniqueness constraints...")
for driver in available_drivers:
    prob += pulp.lpSum([assign_driver[(driver, car)] for car in cars]) <= 1

# Car capacity constraints
print("  - Capacity constraints...")
for car in cars:
    capacity = car_to_capacity[car]
    prob += pulp.lpSum([assign_passenger[(person, car)] for person in all_participants]) <= capacity

# Driver can only drive authorized cars
print("  - Driver authorization constraints...")
for driver in available_drivers:
    allowed_cars = driver_allowed_cars.get(driver, [])
    for car in cars:
        if car not in allowed_cars:
            prob += assign_driver[(driver, car)] == 0

# No male driver with all female passengers
print("  - Gender mixing constraints...")
for driver in available_drivers:
    driver_gender = participant_to_gender.get(driver, 'Unknown')
    if driver_gender == 'M':
        for car in cars:
            # If male driver assigned, at least one male passenger must be in car
            prob += assign_driver[(driver, car)] <= pulp.lpSum([assign_passenger[(person, car)] for person in male_participants if person in all_participants])

# Minibus needs at least 2 leaders (but only 1 needs to be the driver)
print("  - Minibus leader constraints...")
for car in minibus_cars:
    # Count leaders as both passengers and drivers in this car
    leaders_as_passengers = pulp.lpSum([assign_passenger[(leader, car)] for leader in leaders])
    leaders_as_drivers = pulp.lpSum([assign_driver[(leader, car)] for leader in leaders if leader in available_drivers])
    total_leaders_in_car = leaders_as_passengers + leaders_as_drivers
    prob += total_leaders_in_car >= 2

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

# Extract assignments
passenger_assignments = {}
driver_assignments = {}
all_assignments = {}  # Combined view

for person in all_participants:
    for car in cars:
        if assign_passenger[(person, car)].varValue == 1:
            passenger_assignments[person] = car
            all_assignments[person] = ("passenger", car)

for driver in available_drivers:
    for car in cars:
        if assign_driver[(driver, car)].varValue == 1:
            driver_assignments[car] = driver
            all_assignments[driver] = ("driver", car)

# Create car-centric view
car_assignments = {}
for car in cars:
    passengers_in_car = [person for person, assigned_car in passenger_assignments.items() if assigned_car == car]
    if passengers_in_car or car in driver_assignments:
        car_assignments[car] = passengers_in_car

#%% Debug constraint conflicts
print("\n" + "="*60)
print("DEBUGGING CONSTRAINT CONFLICTS")
print("="*60)

# Check specific example: Alice Cornes group with WOLF car
alice_cornes_group = going_people.filter(pl.col('Dorm') == 'Alice Cornes').select('Name').to_series().to_list()
print(f"\nAlice Cornes group members: {alice_cornes_group}")
print(f"Alice Cornes group size: {len(alice_cornes_group)}")

# Check WOLF car capacity
wolf_capacity = car_to_capacity.get('Cornes (WOLF)', 'Not found')
print(f"WOLF car capacity: {wolf_capacity}")

# Check if Alice Cornes can drive WOLF
alice_authorized_cars = driver_allowed_cars.get('Alice Cornes', [])
print(f"Alice Cornes can drive: {alice_authorized_cars}")

# Check leader status
print(f"Alice Cornes is leader: {'Alice Cornes' in leaders}")

# Theoretical best allocation for Alice Cornes + WOLF
if len(alice_cornes_group) >= 4 and wolf_capacity >= 4:
    print(f"\n🎯 THEORETICAL OPTIMAL: Alice Cornes driving 4 of her room group")
    theoretical_score = (4 * 4 * 1000) + (3 * 10000)  # Group bonus + leader bonus
    print(f"   Expected score: {theoretical_score:,} points")
    
    # Check what might prevent this
    print("\n🔍 CONSTRAINT CONFLICTS CHECK:")
    
    # 1. Room group size
    if len(alice_cornes_group) < 5:  # Need Alice + 4 passengers
        print("   ❌ Not enough Alice Cornes group members")
    else:
        print("   ✅ Enough Alice Cornes group members available")
    
    # 2. Authorization constraint  
    if 'Cornes (WOLF)' not in alice_authorized_cars:
        print("   ❌ Alice Cornes not authorized to drive WOLF")
    else:
        print("   ✅ Alice Cornes authorized to drive WOLF")
    
    # 3. Capacity constraint
    if wolf_capacity < 4:
        print(f"   ❌ WOLF capacity too small: {wolf_capacity}")
    else:
        print(f"   ✅ WOLF capacity sufficient: {wolf_capacity}")

# Check what the algorithm actually allocated
print(f"\n📊 ACTUAL ALLOCATION:")
wolf_driver = driver_assignments.get('Cornes (WOLF)', 'No driver')
wolf_passengers = car_assignments.get('Cornes (WOLF)', [])
print(f"   WOLF driver: {wolf_driver}")
print(f"   WOLF passengers: {wolf_passengers}")

if wolf_passengers:
    passenger_groups = {}
    for passenger in wolf_passengers:
        group = participant_to_group.get(passenger, 'Unknown')
        passenger_groups[group] = passenger_groups.get(group, 0) + 1
    
    print(f"   Group breakdown: {passenger_groups}")
    min_group_size = min(passenger_groups.values()) if passenger_groups else 0
    print(f"   Minimum group size: {min_group_size}")
    
    # Calculate actual score for this car
    alice_cornes_count = passenger_groups.get('Alice Cornes', 0)
    if wolf_driver == 'Alice Cornes':
        alice_cornes_count += 1  # Include driver
    
    actual_score = alice_cornes_count * alice_cornes_count * 1000
    if wolf_driver == 'Alice Cornes':
        actual_score += (alice_cornes_count - 1) * 10000  # Leader bonus for passengers
    
    print(f"   Estimated score: {actual_score:,} points")

print("\n" + "="*60)

#%% Display results
print("\n" + "="*60)
print("TRANSPORT ALLOCATION RESULTS")
print("="*60)

total_passengers = 0
for car in sorted(cars):
    if car in car_assignments and car_assignments[car]:
        passengers_in_car = car_assignments[car]
        total_passengers += len(passengers_in_car)
        
        print(f"\n🚗 {car}")
        print(f"   Driver: {driver_assignments.get(car, '❌ NO DRIVER ASSIGNED')}")
        print(f"   Passengers ({len(passengers_in_car)}):")
        
        # Group passengers by their groups for easier reading
        passengers_by_group = {}
        for passenger in passengers_in_car:
            group = participant_to_group.get(passenger, 'Unknown')
            if group not in passengers_by_group:
                passengers_by_group[group] = []
            passengers_by_group[group].append(passenger)
        
        for group, members in passengers_by_group.items():
            leader_count = sum(1 for m in members if m in leaders)
            leader_indicator = f" ({leader_count}L)" if leader_count > 0 else ""
            print(f"     {group}{leader_indicator}: {', '.join(sorted(members))}")

print(f"\nTotal participants assigned: {len(passenger_assignments) + len(driver_assignments)}/{len(all_participants)}")

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

# Check all participants assigned (as passengers or drivers)
unassigned = []
for person in all_participants:
    is_passenger = person in passenger_assignments
    is_driver = any(person == driver for driver in driver_assignments.values())
    
    if not is_passenger and not is_driver:
        unassigned.append(person)

if unassigned:
    print(f"\n❌ UNASSIGNED PARTICIPANTS ({len(unassigned)}):")
    for person in unassigned:
        print(f"  - {person}")
    all_good = False
else:
    print("\n✅ ALL PARTICIPANTS ASSIGNED (as passengers or drivers)")

# Summary
print("\n" + "="*60)
if all_good:
    print("🎉 ALL VALIDATION CHECKS PASSED!")
else:
    print("⚠️  Some validation issues found - review above")

print(f"\nTotal runtime: {time.time() - start_time:.1f} seconds")
print("="*60)
#%%