trunc = lambda x: str(x)[:8]

def make_time_units_string(time):
    # Show the time with appropriate easy-to-read units.
    if time < 0:
        return trunc(time * 1000) + " milliseconds"
    elif time < 60:
        return trunc(time) + " seconds"
    elif time < 3600:
        return trunc(time / 60) + " minutes"
    else:
        return trunc(time / 3600) + " hours"