#%%
import polars as pl
import pulp
def calculate_transport_score(camper, car, campers_df, cars_df):
    score = 50  # Base score
    
    # Get camper's group
    camper_group = campers_df.filter(pl.col('Name') == camper)['Dorm'].item()
    
    # Bonus for being with group members (we'll calculate this in constraints)
    # This is handled by counting group members in same car
    
    # Penalty for middle seats (if car has odd number of seats > 5)
    # car_seats = cars_df.filter(pl.col('Surname & Car Reg') == car)['Seats'].item()
    # if car_seats >= 7:  # Has middle seats
    #     score -= 10  # We'll handle "only if necessary" in constraints
    
    return score

# %%
