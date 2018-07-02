def mapunits(input_len, size='normal'):
    heuristic_map_units = 5 * input_len ** 0.54321

    if size == 'big':
        heuristic_map_units = 4 * (heuristic_map_units)
    elif size == "small":
        heuristic_map_units = 0.25 * (heuristic_map_units)

    return heuristic_map_units
