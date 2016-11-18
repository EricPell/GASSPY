def turb_velocity(x,y,z,dx):
    # identify all cells within a dx of the current cell. if neighboring cells have higher refinement could return more than 6
    # The code solves motion only along cardinal axis. so only worry about cells that can actually send mass into this cell.
    my neighbor_mask = ( abs(dd["x"] - x) <= dx  ) and (abs(dd["y"] - y) <= dx) and (abs(dd["z"]-z) <= dx))
    
    # Am I doing this with heating or turbulence. I would prefer turbulence. Perhaps there should be an option.
    # Or if turbulence is only a heating term (it's not) I could use the same cloudy formula for heating and calculate it myself
    # from the existing velocity field. But turbulence provides an increased collisional rate. 

    #. # calculate relative velocity to all neighboring cells

    #. # Identify which cells are moving in access of the sound speed TOWARD the current cell in it's referece frame

    #. # The energy due to compressive heating is already accounted for in the current temperature of the gas. 
    
    #. # Calculate momentum flux into current

    #. #   
