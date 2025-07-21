#%%
import polars as pl
import pulp
import utils

#%% Read your data
people = pl.read_csv("data/people.csv")
# campers_df = people.filter(pl.col('Year').is_in(['L']).not_())
campers = people.select('Name').to_series().to_list()
cars_df = pl.scan_csv('data/cars.csv'
                ).with_columns(
                    (pl.col('3-point belts (excluding driver)')+1).alias('seats')
                ).collect()
cars = cars_df.select(pl.col('Surname & Car Reg')).to_series().to_list()
drivers = pl.scan_csv('data/cars.csv').select(['Driver 1']).collect().to_series().drop_nulls().to_list() + \
          pl.scan_csv('data/cars.csv').select(['Driver 2']).collect().to_series().drop_nulls().to_list() + \
          pl.scan_csv('data/cars.csv').select(['Driver 3']).collect().to_series().drop_nulls().to_list() + \
          pl.scan_csv('data/cars.csv').select(['Driver 4']).collect().to_series().drop_nulls().to_list()
drivers = list(set(drivers))

#%% Create the optimization problem
prob = pulp.LpProblem("Transport_Allocation", pulp.LpMaximize)

#%% Assign Camper and Driver

# Camper assigned to car
assign_camper = {}
for c in campers:
    for car in cars:
        assign_camper[(c, car)] = pulp.LpVariable(f"camper_{c}_{car}", cat='Binary')

# Driver assigned to car
assign_driver = {}
for d in drivers:
    for car in cars:
        assign_driver[(d, car)] = pulp.LpVariable(f"driver_{d}_{car}", cat='Binary')

#%% Calculate scores
scores = {}
for c in campers:
    for car in cars:
        scores[(c, car)] = 50

# %% Add bonus for group cohesion

# Base satisfaction
prob += pulp.lpSum([scores[(c, car)] * assign_camper[(c, car)] 
                   for c in campers for car in cars])

group_together = {}
groups = people.select('Dorm').unique().to_series().to_list()
for group in groups:
    group_members = people.filter(pl.col('Dorm') == group)['Name'].to_list()
    
    for car in cars:
        # Bonus for each pair of group members in same car
        for i, member1 in enumerate(group_members):
            for j, member2 in enumerate(group_members):
                # prob += 20 * assign_camper[(member1, car)] * assign_camper[(member2, car)]
                if i < j:
                    var_name = f"together_{member1}_{member2}_{car}".replace(' ', '_').replace('\\', '')
                    group_together[(member1, member2, car)] = pulp.LpVariable(var_name, cat='Binary')

for group in groups:
    group_members = people.filter(pl.col('Dorm') == group)['Name'].to_list()

    for car in cars:
        for i, member1 in enumerate(group_members):
            for j, member2 in enumerate(group_members):
                if i < j:
                    key = (member1,member2,car)
                    if key in group_together:                                  
                    # Constraints to link this to the assignment variables
                        prob += group_together[key] <= assign_camper[(member1, car)]
                        prob += group_together[key] <= assign_camper[(member2, car)]
                        prob += group_together[key] >= (
                            assign_camper[(member1, car)] + assign_camper[(member2, car)] - 1
                        )

            # Add to objective
            prob += pulp.lpSum([20 * group_together[(m1, m2, car)]
                            for group in groups 
                            for car in cars
                            for i, m1 in enumerate(people.filter(pl.col('Dorm') == group).select('Name').to_series().to_list())
                            for m2 in people.filter(pl.col('Dorm') == group).select('Name').to_series().to_list()[i+1:]])

#%%
# Each camper in exactly one car
for c in campers:
    prob += pulp.lpSum([assign_camper[(c, car)] for car in cars]) == 1

# Each car has exactly one driver
for car in cars:
    prob += pulp.lpSum([assign_driver[(d, car)] for d in drivers]) == 1

# Car capacity (seats - 1 for driver)
for car in cars:
    capacity = cars_df.filter(pl.col('Surname & Car Reg') == car).select('seats').to_series().item() - 1
    prob += pulp.lpSum([assign_camper[(c, car)] for c in campers]) <= capacity

# Driver can only drive cars they're insured for
for d in drivers:
    for car in cars:
        allowed_drivers = cars_df.filter(pl.col('Surname & Car Reg')==car
                            ).select([f'Driver {x}' for x in range(1,7)]
                            ).transpose().select('column_0').to_series(
                            ).drop_nulls().to_list()
        if d not in allowed_drivers:
            prob += assign_driver[(d, car)] == 0

# No male driver with all female passengers
for d in drivers:
    driver_gender = people.filter(pl.col('Name') == d).select('Sex').item()
    if driver_gender == 'M':
        for car in cars:
            # If male driver assigned to this car, at least one male camper must be too
            male_campers = people.filter(pl.col('Sex') == 'M').select('Name').to_series().drop_nulls().to_list()
            prob += assign_driver[(d, car)] <= pulp.lpSum([assign_camper[(mc, car)] for mc in male_campers])

# Minibus needs second leader
minibus_cars = cars_df.filter(pl.col('Surname & Car Reg').str.contains('Minibus')).select('Surname & Car Reg').to_series().to_list()
leaders = people.filter(pl.col('Year') == "L").select('Name').to_series().to_list()

for car in minibus_cars:
    prob += pulp.lpSum([assign_camper[(leader, car)] for leader in leaders]) >= 2

# Leaders prefer to go with their group
leader_with_group = {}
for leader in leaders:
    leader_group = people.filter(pl.col('Name') == leader).select('Dorm').to_series().item()
    group_members = people.filter(pl.col('Dorm') == leader_group).select('Name').to_series().to_list()

    for car in cars:
        for member in group_members:
            if member != leader:
                # Binary variable: 1 if both leader and member in same car
                var_name = f"leader_with_{leader}_{member}_{car}".replace(' ', '_').replace('(', '').replace(')', '')
                leader_with_group[(leader, member, car)] = pulp.LpVariable(var_name, cat='Binary')
                
                # Linking constraints
                prob += leader_with_group[(leader, member, car)] <= assign_camper[(leader, car)]
                prob += leader_with_group[(leader, member, car)] <= assign_camper[(member, car)]
                prob += leader_with_group[(leader, member, car)] >= (
                    assign_camper[(leader, car)] + assign_camper[(member, car)] - 1
                )

    # Add to objective function
    prob += pulp.lpSum([15 * leader_with_group[(leader, member, car)] 
                    for leader in leaders
                    for car in cars
                    for leader_group in [people.filter(pl.col('Name') == leader).select('Dorm').to_series().item()]
                    for member in people.filter(pl.col('Dorm') == leader_group).select('Name').to_series().to_list()
                    if member != leader and (leader, member, car) in leader_with_group])
# %%
# Solve the optimization problem
prob.solve()

# Check if solution was found
print("Status:", pulp.LpStatus[prob.status])

if prob.status == pulp.LpStatusOptimal:
    print("Optimal solution found!")
    print(f"Total score: {pulp.value(prob.objective)}")
else:
    print("No optimal solution found")
    print("Status code:", prob.status)
#%%
# Extract camper assignments
camper_assignments = {}
car_assignments = {}

for c in campers:
    for car in cars:
        if assign_camper[(c, car)].varValue == 1:
            camper_assignments[c] = car
            if car not in car_assignments:
                car_assignments[car] = []
            car_assignments[car].append(c)

# Extract driver assignments
driver_assignments = {}
for d in drivers:
    for car in cars:
        if assign_driver[(d, car)].varValue == 1:
            driver_assignments[car] = d
#%%
print("\n=== TRANSPORT ALLOCATION RESULTS ===")
for car in cars:
    if car in car_assignments:
        print(f"\n{car}:")
        print(f"  Driver: {driver_assignments.get(car, 'No driver assigned')}")
        print(f"  Passengers ({len(car_assignments[car])}):")
        
        # Group passengers by their groups for easier reading
        passengers_by_group = {}
        for passenger in car_assignments[car]:
            group = people.filter(pl.col('Name') == passenger).select('Dorm').to_series().item()
            if group not in passengers_by_group:
                passengers_by_group[group] = []
            passengers_by_group[group].append(passenger)
        
        for group, members in passengers_by_group.items():
            print(f"    {group}: {', '.join(members)}")
#%%
print("\n=== VALIDATION ===")

# Check capacity constraints
for car in car_assignments:
    capacity = cars_df.filter(pl.col('Surname & Car Reg') == car)['seats'].item() - 1  # minus driver
    actual = len(car_assignments[car])
    print(f"{car}: {actual}/{capacity} passengers")
    if actual > capacity:
        print(f"  ⚠️  OVER CAPACITY!")

# Check group cohesion
groups = people.select('Dorm').unique().to_series().to_list()
for group in groups:
    group_members = people.filter(pl.col('Dorm') == group).select('Name').to_series().to_list()
    cars_used = set()
    for member in group_members:
        if member in camper_assignments:
            cars_used.add(camper_assignments[member])
    
    print(f"Group {group}: spread across {len(cars_used)} cars")
    if len(cars_used) > 1:
        print(f"  Cars: {list(cars_used)}")

# Check minibus leader requirement
minibus_cars = cars_df.filter(pl.col('Surname & Car Reg').str.contains('Minibus')).select('Surname & Car Reg').to_series().to_list()
for car in minibus_cars:
    if car in car_assignments:
        leaders_in_car = [p for p in car_assignments[car] if p in leaders]
        print(f"Minibus {car}: {len(leaders_in_car)} leaders")
        if len(leaders_in_car) < 2:
            print(f"  ⚠️  NEEDS MORE LEADERS!")
#%%