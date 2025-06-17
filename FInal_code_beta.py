#IMPORTED LIBRARIES
import numpy as np #matrix lib  
import matplotlib.pyplot as plt #plotting lib
import os  #file management lib
import random  #random generationlib
import time #timing gates lib
import pandas as pd
import sys #skeleton
import math #rounding down and up 
import json #saving the settings
import trimesh
import csv
import os 
from shapely import affinity
from shapely.validation import make_valid 
from shapely.geometry import LineString, Polygon, MultiPolygon, box
from shapely.ops import unary_union
import svgwrite
from svgwrite.path import Path
import shutil  #file management lib
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QSpinBox, QDoubleSpinBox, QVBoxLayout, QSizePolicy,
    QHBoxLayout, QPushButton, QTabWidget, QCheckBox, QLineEdit, QStackedWidget,
    QFileDialog, QFormLayout, QGroupBox, QMessageBox, QScrollArea, QGridLayout, QSlider
)
from PyQt5.QtGui import QFont
from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtCore import Qt, QByteArray

TILE_SIZE = 3
#########################################################################################################################################################
# Loop check fucntion - no internal reinforced unconnected structure within a reinforced structure 
def create_simulation_tile(up=0, right=0, down=0, left=0):
    tile = np.zeros((3, 3), dtype=int)
    tile[0, 1] = up
    tile[1, 2] = right
    tile[2, 1] = down
    tile[1, 0] = left
    return tile

def merged_grain_check(grid, x, y, max_steps=400):
    deltas = {
        'up': (-1, 0), 'down': (1, 0),
        'left': (0, -1), 'right': (0, 1)
    }
    wall_map = {
        'up': (0, 1), 'down': (2, 1),
        'left': (1, 0), 'right': (1, 2)
    }
    reverse_map = {
        'up': 'down', 'down': 'up',
        'left': 'right', 'right': 'left'
    }

    def tile_sum(tile):
        return int(np.sum(tile)) if tile is not None else 0

    def is_type2(tile, required):
        return tile_sum(tile) == 2 and all(tile[r, c] == 1 for r, c in required)

    def is_type3(tile, forbidden):
        return tile_sum(tile) == 4 and all(tile[r, c] == 0 for r, c in forbidden)

    H, W = grid.shape[:2]
    if x == 0 or y == 0:
        return False

    top = grid[y - 1, x]
    left = grid[y, x - 1]
    diag = grid[y - 1, x - 1]

    # === 4-Tile Diagonal Type-2 Grain Check
    if all(t is not None for t in [top, left, diag]):
        if is_type2(top, [(1, 0), (2, 1)]) and \
           is_type2(left, [(0, 1), (1, 2)]) and \
           is_type2(diag, [(1, 2), (2, 1)]):
            #print(f"[❌ Diagonal loop detected around ({x},{y})]")
            return True

    # === Extended 6-Tile Diagonal Vertical Type-3 Bridge
    if x >= 2 and y >= 2:
        tl = grid[y - 2, x - 2]
        t3 = grid[y - 2, x - 1]
        tr = grid[y - 2, x]
        l3 = grid[y - 1, x - 2]
        if all(t is not None for t in [tl, t3, tr, l3]):
            if is_type2(tl, [(1, 2), (2, 1)]) and \
               is_type2(tr, [(1, 0), (2, 1)]) and \
               is_type3(t3, [(1, 0)]) and \
               is_type3(l3, [(1, 2)]):
                #print(f"[❌ Extended vertical diagonal loop around ({x},{y})]")
                return True

    # === Extended 6-Tile Diagonal Horizontal Type-3 Bridge
    if x >= 2 and y >= 2:
        tr = grid[y - 2, x]
        r3 = grid[y - 1, x + 1] if x + 1 < W else None
        rt = grid[y - 2, x + 1] if x + 1 < W else None
        if all(t is not None for t in [tr, r3, rt]):
            if is_type2(tr, [(1, 0), (2, 1)]) and \
               is_type2(grid[y - 1, x - 1], [(0, 1), (1, 2)]) and \
               is_type3(r3, [(0, 1)]) and \
               is_type3(rt, [(2, 1)]):
                #print(f"[❌ Extended horizontal diagonal loop around ({x},{y})]")
                return True

    # === Horizontal Type-3 Trap Check
    if x >= 2:
        left_1 = grid[y, x - 1]
        left_2 = grid[y, x - 2]
        if all(t is not None for t in [left_1, left_2]):
            if tile_sum(left_1) == 4 and tile_sum(left_2) == 4:
                #print(f"[❌ Horizontal type-3 trap detected at ({x},{y})]")
                return True

    # === Vertical Type-3 Trap Check
    if y >= 2:
        top_1 = grid[y - 1, x]
        top_2 = grid[y - 2, x]
        if all(t is not None for t in [top_1, top_2]):
            if tile_sum(top_1) == 4 and tile_sum(top_2) == 4:
                #print(f"[❌ Vertical type-3 trap detected at ({x},{y})]")
                return True

    # === Flood Fill Walk Check (Simulation)
    top_tile = grid[y - 1, x] if y - 1 >= 0 else None
    left_tile = grid[y, x - 1] if x - 1 >= 0 else None

    simulated_tile = create_simulation_tile(up=1, left=1)
    grid_backup = grid[y, x]
    grid[y, x] = simulated_tile

    stack = []
    if top_tile is not None and top_tile[2, 1] == 1:
        stack.append((y - 1, x))
    if left_tile is not None and left_tile[1, 2] == 1:
        stack.append((y, x - 1))

    visited = set()
    steps = 0
    escaped = False

    while stack and steps < max_steps:
        cy, cx = stack.pop()

        if (cy, cx) == (y, x):
            if steps > 3:
                #print(f"[❌ Loop Detected] Return to ({x},{y}) after {steps} steps")
                break
            continue

        if (cy, cx) in visited:
            continue
        visited.add((cy, cx))

        for direction in deltas:
            dy, dx = deltas[direction]
            ny, nx = cy + dy, cx + dx

            if not (0 <= ny < H and 0 <= nx < W):
                if grid[cy, cx][wall_map[direction]] == 1:
                    if (direction == 'up' and cy == 0) or \
                       (direction == 'down' and cy == H - 1) or \
                       (direction == 'left' and cx == 0) or \
                       (direction == 'right' and cx == W - 1):
                        #print(f"[✅ Escaped] via boundary at ({cy},{cx})")
                        escaped = True
                        break
                continue

            neighbor = grid[ny, nx]
            if ((ny > y and nx == x) or (nx > x and ny == y)) and \
               neighbor is not None and np.sum(neighbor) == 0:
                #print(f"[✅ Escaped] to unassigned at ({ny},{nx})")
                escaped = True
                break

            if neighbor is not None and \
               grid[cy, cx][wall_map[direction]] == 1 and \
               neighbor[wall_map[reverse_map[direction]]] == 1 and \
               (ny, nx) not in visited:
                stack.append((ny, nx))

        if escaped:
            break
        steps += 1

    grid[y, x] = grid_backup
    return not escaped
#checking the position of the tile in the grid, assigning boolean values to relevant cases left and top tile connectivity, grid boundary cases
def generation_demo_1_grid_constraints(x, y, grid):
    """Determine tile constraints based on grid boundaries and ensure correct edge tile placement."""
    constraints = {
        "x": x,
        "y": y,
        "boundary_left": x == 0,
        "boundary_right": x == GRID_WIDTH - 1,
        "boundary_top": y == 0,
        "boundary_bottom": y == GRID_HEIGHT - 1,
        "requires_left": False,
        "requires_top": False,
        "circle": False,
        "active_connections": [0, 0, 0, 0],  
    }
    center = 1
    constraints["disallow_tile_types"] = set()   
    if x > 0:
        if grid[y, x-1][center, 2] == 1:  # If left tile has right connection
            constraints["requires_left"] = True
            constraints["active_connections"][0] = 1  # left wall connection 
    if y > 0:  
        if  grid[y-1, x][2, center] == 1:  # If top tile has bottom connection
            constraints["requires_top"] = True
            constraints["active_connections"][3] = 1  # top wall connection
    if constraints["requires_left"] and constraints["requires_top"]:
        constraints["circle"] = merged_grain_check(grid, x, y, max_steps=400)
    else:
        constraints["circle"] =False
    return constraints  
############################## BREAK FOR THE COUNTING ANALYSIS PER PHASE TO CORRECT FOR RANDOM GENERATION ON SMAL SCALE OBJECT ####################################
#basicaly 4 dictironaries comparing what is the expected current state and fixing the wieghts if it is of from the expected, also adjusting the stored output
def adjust_probabilities(frequency_targets, placed_counts, total_tiles_placed, tile_possibility_matrix, mode, weight):
    tile_type_to_index = {0: 0, 2: 1, 3: 2, 4: 3}
    coordination_weight = {0: 0.0, 2: 2.0, 3: 3.0, 4: 4.0}
    # Only consider tile types that are currently possible
    desired_counts = {
        int(tile_type): frequency_targets[tile_type] * total_tiles_placed
        for tile_type in frequency_targets
        if tile_possibility_matrix[tile_type_to_index[int(tile_type)], 0] == 1
    }
    diffs = {
        tile_type: desired_counts[tile_type] - placed_counts.get(tile_type, 0)
        for tile_type in desired_counts
    }
    adjusted_weights = {}
    for tile_type, diff in diffs.items():
        expected_count = max(frequency_targets.get(tile_type, 0.001) * total_tiles_placed, 1e-6)
        impact = coordination_weight.get(tile_type, 1.0)
        scaled_diff = diff * impact
        relative_diff = scaled_diff / expected_count

        # NEW: In transition mode, heavily penalize tile type 3 if overused
        if mode == "transition" and tile_type == 3 and diff <= 0:
            adjusted_weights[tile_type] = 0.0001
            continue  # Skip to next tile
        if diff <= 0 and tile_type in [0, 4]:  # overused tiles are banned
            adjusted_weights[tile_type] = 0.0
        else:
            raw_adjustment = 1 + weight * relative_diff
            if tile_type == 0:
                raw_adjustment *= 1.5  # Boost recovery for tile 0
            capped = min(max(raw_adjustment, 0.01), 6.0 if tile_type in [0, 4] else 2.0)
            adjusted_weights[tile_type] = capped
    for tile_type in [0, 2, 3, 4]:
        if tile_possibility_matrix[tile_type_to_index[tile_type], 0] == 0:
            adjusted_weights[tile_type] = 0.0
    total_weight = sum(adjusted_weights.values())
    if total_weight == 0:
        #print("[WARNING] Total adjusted weight is zero — fallback to uniform over allowed.")
        allowed = [t for t in [0, 2, 3, 4] if tile_possibility_matrix[tile_type_to_index[t], 0] == 1]
        return {t: 1.0 / len(allowed) for t in allowed}
    return {
        tile_type: adjusted_weights[tile_type] / total_weight
        for tile_type in adjusted_weights
    }    
############################## CONTINUATION OF THE ACTUAL TILE CHOICE #############################################################################    
#checking the tile options and using the frequencies to randomly generate the tile that can fit 
def generation_demo_2_filter_tile_options(
    constraints, grid, x, y, phase_map, blended_phases_map, blend_mask,
    tile_matrix, frequencies, phase_enforced_rules,
    blend_weight_map=None, direction_map=None, tiles_per_phase_x=None, tiles_per_phase_y=None
):
    possible_R = [0, 2, 3, 4]

    # --- STEP 1: PROPAGATION RULES (always enforced) ---
    if tile_matrix[0, 0] == 0:
        possible_R = [t for t in possible_R if t != 0]
    if tile_matrix[1, 0] == 0:
        possible_R = [t for t in possible_R if t != 2]
    if tile_matrix[2, 0] == 0:
        possible_R = [t for t in possible_R if t != 3]
    if tile_matrix[3, 0] == 0:
        possible_R = [t for t in possible_R if t != 4]

    if constraints["requires_left"] or constraints["requires_top"]:
        possible_R = [t for t in possible_R if t != 0]

    if constraints["circle"]:
        possible_R = [t for t in possible_R if t != 2]

    if not constraints["requires_left"] and not constraints["boundary_left"] and \
    not constraints["requires_top"] and not constraints["boundary_top"]:
        possible_R = [t for t in possible_R if t != 4]

    if not (constraints["requires_left"] or constraints["requires_top"]) and \
    not (constraints["boundary_left"] or constraints["boundary_top"]):
        possible_R = [t for t in possible_R if t != 3]

    phase = phase_map[y][x]
    allowed_types = evaluate_enforced_rules(
        x, y, grid, constraints, phase, phase_enforced_rules,
        blend_mask=blend_mask,
        blended_phases_map=blended_phases_map,
        blend_weight_map=blend_weight_map,
        tiles_per_phase_x=tiles_per_phase_x,
        tiles_per_phase_y=tiles_per_phase_y,
        direction_map=direction_map,
        phase_map=phase_map
    )
    possible_R = [r for r in possible_R if r in allowed_types]

    # --- STEP 3: Frequency-based choice ---
    if not possible_R:
        #print(f"[Warning] No valid tile types at ({x},{y}) — fallback to tile 0")
        return 0

    adjusted_frequencies = {t: frequencies.get(t, 1.0) for t in possible_R}
    total = sum(adjusted_frequencies.values())
    total = sum(adjusted_frequencies.values())

    if total > 0:
        probabilities = [adjusted_frequencies[t] / total for t in possible_R]
    else:
        fallback = possible_R 
        probabilities = [1.0 / len(fallback) for _ in fallback]

    return random.choices(possible_R, probabilities)[0]

# === Helper Function: Evaluate Enforced Rules per Phase and Neighbor Tile ===
def evaluate_enforced_rules(
    x, y, grid, constraints, phase, phase_rules,
    blend_mask=None, blended_phases_map=None, blend_weight_map=None,
    tiles_per_phase_x=None, tiles_per_phase_y=None, direction_map=None,  phase_map=None
):
    allowed_types = {0, 2, 3, 4}
    left_tile = grid[y, x - 1] if x > 0 else None
    left_tile_key = get_tile_key(left_tile, np.sum(left_tile)) if left_tile is not None else None
    current_rules = phase_rules.get(phase, {})

    def maybe_discard(tile_type, rule_key_check, condition, decay_factor=1.0):
        """Handles normal or decayed rule-based tile bans."""
        if not condition:
            return
        # If in blend region, apply decay based on blending weight
        if blend_mask is not None and blend_mask[y, x]:
            if blend_weight_map and blended_phases_map and direction_map:
                weights = get_rule_blend_weights(
                    x, y,
                    neighbor_phases=blended_phases_map[y][x],
                    phase_map=phase_map, 
                    tiles_per_phase_x=tiles_per_phase_x,
                    tiles_per_phase_y=tiles_per_phase_y,
                    direction_map=direction_map
                )
                for neighbor_phase, weight in weights.items():
                    neighbor_rules = phase_rules.get(neighbor_phase, {})
                    if neighbor_rules.get(rule_key_check, False):
                        if random.random() < weight * decay_factor:
                            allowed_types.discard(tile_type)
                            break
        else:
            # Normal strict rule enforcement
            if current_rules.get(rule_key_check, False):
                allowed_types.discard(tile_type)

    # === Apply Tile-Specific Rules ===
    maybe_discard(
        tile_type=3,
        rule_key_check="disallow_tile_3_noB_after_tile_3_noB",
        condition=left_tile_key == "tile_3_noB" and not constraints["requires_top"]
    )

    maybe_discard(
        tile_type=3,
        rule_key_check="disallow_tile_3_noL_after_tile_3_noR",
        condition=left_tile_key == "tile_3_noR"
    )

    maybe_discard(
        tile_type=4,
        rule_key_check="disallow_tile_4_after_tile_4",
        condition=left_tile_key == "tile_4"
    )

    maybe_discard(
        tile_type=0,
        rule_key_check="disallow_tile_0_after_tile_0",
        condition=left_tile_key == "tile_0"
    )

    return allowed_types

def get_rule_blend_weights(x, y, neighbor_phases, phase_map, tiles_per_phase_x, tiles_per_phase_y, direction_map):
    VALID_TILES = {0, 2, 3, 4}
    directions = direction_map[y][x] if isinstance(direction_map[y][x], (list, set)) else {direction_map[y][x]}
    direction = next(iter(directions)) if directions else None

    current_phase = phase_map[y][x]
    region_min_x = (x // tiles_per_phase_x) * tiles_per_phase_x
    region_max_x = region_min_x + tiles_per_phase_x - 1
    region_min_y = (y // tiles_per_phase_y) * tiles_per_phase_y
    region_max_y = region_min_y + tiles_per_phase_y - 1

    def compute_alpha(min_y, max_y, min_x, max_x, direction):
        if direction == "top":
            return (y - min_y) / max((max_y - min_y), 1)
        elif direction == "bottom":
            return (max_y - y) / max((max_y - min_y), 1)
        elif direction == "left":
            return (x - min_x) / max((max_x - min_x), 1)
        elif direction == "right":
            return (max_x - x) / max((max_x - min_x), 1)
        else:
            norm_y = (y - min_y) / max((max_y - min_y), 1)
            norm_x = (x - min_x) / max((max_x - min_x), 1)
            return 0.5 * (norm_x + norm_y)

    alpha = compute_alpha(region_min_y, region_max_y, region_min_x, region_max_x, direction)

    # === Assign weights for 2-phase blends ===
    if len(neighbor_phases) == 2:
        pid_from = current_phase
        pid_to = [pid for pid in neighbor_phases if pid != current_phase][0]

        if direction in {"bottom", "right"}:
            weights = {
                pid_from: 1.0 - alpha,
                pid_to: alpha
            }
        elif direction in {"top", "left"}:
            weights = {
                pid_to: alpha,
                pid_from: 1.0 - alpha
            }
        else:
            weights = {pid: 1.0 / len(neighbor_phases) for pid in neighbor_phases}

    # === Assign weights for 3-phase corner blends ===
    elif len(neighbor_phases) == 3:
        pid_main = current_phase
        pid_neighbors = [pid for pid in neighbor_phases if pid != pid_main]
        pid_top_bottom, pid_left_right = None, None

        for pid in pid_neighbors:
            if y > 0 and phase_map[y - 1][x] == pid:
                pid_top_bottom = pid
            if y < len(phase_map) - 1 and phase_map[y + 1][x] == pid:
                pid_top_bottom = pid
            if x > 0 and phase_map[y][x - 1] == pid:
                pid_left_right = pid
            if x < len(phase_map[0]) - 1 and phase_map[y][x + 1] == pid:
                pid_left_right = pid

        beta = (x - region_min_x) / max((region_max_x - region_min_x), 1)
        alpha = (y - region_min_y) / max((region_max_y - region_min_y), 1)

        main_weight = max(0.0, 1.0 - alpha * 0.4 - beta * 0.4)
        weights = {pid_main: main_weight}
        if pid_top_bottom is not None:
            weights[pid_top_bottom] = alpha * 0.4
        if pid_left_right is not None:
            weights[pid_left_right] = beta * 0.4

    else:
        weights = {pid: 1.0 / len(neighbor_phases) for pid in neighbor_phases}

    return weights

def generation_demo_3_assign_remaining_connections(
    grid, x, y, chosen_sum, constraints, phase_rules, phase,
    blend_mask=None, blended_phases_map=None, blend_weight_map=None
):
    left_tile = grid[y, x - 1] if x > 0 else None
    left_tile_key = get_tile_key(left_tile, np.sum(left_tile)) if left_tile is not None else None
    active_connections = constraints["active_connections"].copy()
    remaining_connections = chosen_sum - sum(active_connections)
    valid_sides = [0, 1, 2, 3]  # L, B, R, T
    current_rules = phase_rules.get(phase, {})

    if not constraints["boundary_left"]:
        valid_sides.remove(0)
    if not constraints["boundary_top"]:
        valid_sides.remove(3)

    # === Helper to maybe prevent/discourage connections ===
    def maybe_force_connection(side, condition, decay_factor=1.0):
        nonlocal remaining_connections
        if condition:
            if blend_mask is not None and blend_mask[y, x]:
                for neighbor_phase in blended_phases_map[y][x]:
                    neighbor_rules = phase_rules.get(neighbor_phase, {})
                    weight = blend_weight_map[y][x].get(neighbor_phase, 0) if blend_weight_map else 0
                    if neighbor_rules.get(condition, False) and random.random() < weight * decay_factor:
                        if active_connections[side] == 0:
                            active_connections[side] = 1
                            remaining_connections -= 1
                            if side in valid_sides:
                                valid_sides.remove(side)
                            break
            else:
                if current_rules.get(condition, False) and active_connections[side] == 0:
                    active_connections[side] = 1
                    remaining_connections -= 1
                    if side in valid_sides:
                        valid_sides.remove(side)

    def maybe_block_connection(side, condition, decay_factor=1.0):
        if condition:
            if blend_mask is not None and blend_mask[y, x]:
                for neighbor_phase in blended_phases_map[y][x]:
                    neighbor_rules = phase_rules.get(neighbor_phase, {})
                    weight = blend_weight_map[y][x].get(neighbor_phase, 0) if blend_weight_map else 0
                    if neighbor_rules.get(condition, False) and random.random() < weight * decay_factor:
                        if side in valid_sides:
                            valid_sides.remove(side)
                            break
            else:
                if current_rules.get(condition, False) and side in valid_sides:
                    valid_sides.remove(side)

    # === Apply enforced/decayed rule logic ===
    if chosen_sum == 3:
        maybe_force_connection(1, "disallow_tile_3_noB_after_tile_3_noB")

    if chosen_sum == 2:
        if left_tile_key == "tile_2_LR":
            maybe_block_connection(2, "disallow_tile_2_LR_after_tile_2_LR")
        elif left_tile_key == "tile_2_LB":
            maybe_block_connection(2, "disallow_tile_2_RT_after_tile_2_LB")

    # === Assign remaining sides randomly ===
    if remaining_connections > 0 and valid_sides:
        selected = random.sample(valid_sides, min(remaining_connections, len(valid_sides)))
        for side in selected:
            if active_connections[side] == 0:
                active_connections[side] = 1
                remaining_connections -= 1

    return active_connections

#filling in the tile once we know the wall connections- overcorrection/ sanity check to fill center if applicable and reasing a tile type and count the connections if something did managae to cruch in between 
def generation_demo_4_construct_tile(active_connections, chosen_sum):
    tile = np.zeros((TILE_SIZE, TILE_SIZE), dtype=int)
    center = 1
    # Apply wall connections
    if active_connections[0]:  # Left
        tile[center, 0] = 1
    if active_connections[1]:  # Bottom
        tile[-1, center] = 1
    if active_connections[2]:  # Right
        tile[center, -1] = 1
    if active_connections[3]:  # Top
        tile[0, center] = 1       
    chosen_sum = sum(active_connections);
    # Ensure proper filling
    if sum(active_connections) == 2:
        if not ((active_connections[0] and active_connections[1]) or 
                (active_connections[1] and active_connections[2]) or 
                (active_connections[2] and active_connections[3]) or 
                (active_connections[3] and active_connections[0])):
            center = (TILE_SIZE ) // 2 
        if active_connections[0] and active_connections[2]:  # Left & Right
            tile[center, :] = 1
        elif active_connections[1] and active_connections[3]:  # Top & Bottom
            tile[:, center] = 1
        else:
            # Handle diagonal connection stepwise
            non_zero_cells = [(center, 0) if active_connections[0] else None,
                              (TILE_SIZE-1, center) if active_connections[1] else None,
                              (center, TILE_SIZE-1) if active_connections[2] else None,
                              (0, center) if active_connections[3] else None]

            non_zero_cells = [cell for cell in non_zero_cells if cell is not None]
            non_zero_cells.sort()  # Sort by x-coordinate to find leftmost cell
            start = non_zero_cells[0]
            end = non_zero_cells[1]
            # Move stepwise from start to end
            x, y = start
            while (x, y) != end:
                tile[x, y] = 1
                if y < end[1]: y += 1  # Move right
                elif y > end[1]: y -= 1  # Move left
                if x < end[0]: x += 1  # Move down
                elif x > end[0]: x -= 1  # Move up
            tile[end] = 1  # Ensure end cell is marked
    elif sum(active_connections) >= 3:
        tile[center, center] = 1  # Connect all walls via center if 3+ walls exist
        for i, conn in enumerate(active_connections):
            if conn:
                if i == 0: tile[center, :center+1] = 1
                elif i == 1: tile[center:, center] = 1
                elif i == 2: tile[center, center:] = 1
                elif i == 3: tile[:center+1, center] = 1           
    return tile, chosen_sum

def generate_boundaries(saved_matrix, grid_width, grid_height, add_horizontal=True, add_vertical=True):
    new_matrix = []

    padded_width = grid_width
    for y in range(grid_height):
        row = saved_matrix[y]

        if add_vertical:
            padded_width = grid_width+ 2
            # LEFT wall
            right_neighbour = row[0]
            active_connections = [0, 1, 0, 1]
            if right_neighbour.get("active_connections", [0, 0, 0, 0])[0]:
                active_connections[2] = 1
            tile, chosen_sum = generation_demo_4_construct_tile(active_connections, sum(active_connections))
            left_tile_new = {"tile": tile, "active_connections": active_connections.copy(), "sum": chosen_sum}

            # RIGHT wall
            left_neighbour = row[-1]
            active_connections = [0, 1, 0, 1]
            if left_neighbour.get("active_connections", [0, 0, 0, 0])[2]:
                active_connections[0] = 1
            tile, chosen_sum = generation_demo_4_construct_tile(active_connections, sum(active_connections))
            right_tile_new = {"tile": tile, "active_connections": active_connections.copy(), "sum": chosen_sum}

            new_matrix.append([left_tile_new] + row + [right_tile_new])
        else:
            new_matrix.append(row.copy())
    top_row = []
    bottom_row = []
    if add_horizontal:
        # ---- TOP ROW ----
        for x in range(padded_width):
            below_tile = new_matrix[0][x] if new_matrix else None
            active_connections = [0, 0, 0, 0]  # l, b, r, t
            is_left_edge = (x == 0)
            is_right_edge = (x == padded_width - 1)
            if is_left_edge & add_vertical:
                 active_connections[2] = 1  # right
            elif is_right_edge& add_vertical:
                active_connections[0] = 1  # left
            else:
                active_connections[0] = 1  # left
                active_connections[2] = 1  # right
            if below_tile and below_tile.get("active_connections", [0, 0, 0, 0])[3]:
                active_connections[1] = 1  # bottom
            tile, chosen_sum = generation_demo_4_construct_tile(active_connections.copy(), sum(active_connections))
            top_row.append({
                "tile": tile,
                "active_connections": active_connections,
                "sum": chosen_sum
            })

        # ---- BOTTOM ROW ----
        for x in range(padded_width):
            above_tile = new_matrix[-1][x]
            active_connections = [0, 0, 0, 0]
            is_left_edge = (x == 0)
            is_right_edge = (x == padded_width - 1)
            if is_left_edge & add_vertical:
                 active_connections[2] = 1  # right
            elif is_right_edge& add_vertical:
                active_connections[0] = 1  # left
            else:
                active_connections[0] = 1  # left
                active_connections[2] = 1  # right
            if above_tile.get("active_connections", [0, 0, 0, 0])[1]:
                active_connections[3] = 1  # top
            tile, chosen_sum = generation_demo_4_construct_tile(active_connections.copy(), sum(active_connections))
            bottom_row.append({
                "tile": tile,
                "active_connections": active_connections,
                "sum": chosen_sum
            })
        new_matrix = [top_row] + new_matrix + [bottom_row]
    return new_matrix

#######################################################################################################################################################
#Functional functions call for the check phase location      
def clean_tile_frequencies(freq_dict, valid_tiles={0, 2, 3, 4}):
    return {
        int(k): float(v)
        for k, v in freq_dict.items()
        if str(k).isdigit() and int(k) in valid_tiles
    }
        
def generate_phase_map(grid_width, grid_height, layout_rows, layout_cols):
    phase_map = np.full((grid_height, grid_width), -1, dtype=int)
    row_block_height = grid_height // layout_rows
    col_block_width = grid_width // layout_cols
    phase_counter = 0
    for row_block in range(layout_rows):
        for col_block in range(layout_cols):
            start_y = row_block * row_block_height
            end_y = start_y + row_block_height
            start_x = col_block * col_block_width
            end_x = start_x + col_block_width
            phase_map[start_y:end_y, start_x:end_x] = phase_counter
            phase_counter += 1
    return phase_map  

def compute_transition_layer(phase_rows, phase_cols, x_shrink, y_shrink):
    margins = {
        "top": {}, "bottom": {}, "left": {}, "right": {}
    }
    for r in range(phase_rows):
        for c in range(phase_cols):
            key = f"{r},{c}"
            if r > 0:
                margins["top"][key] = y_shrink
                up_key = f"{r-1},{c}"
                margins["bottom"][up_key] = y_shrink
            else:
                margins["top"][key] = 0

            if r < phase_rows - 1:
                margins["bottom"][key] = y_shrink
                down_key = f"{r+1},{c}"
                margins["top"][down_key] = y_shrink
            else:
                margins["bottom"][key] = 0
            if c > 0:
                margins["left"][key] = x_shrink
                left_key = f"{r},{c-1}"
                margins["right"][left_key] = x_shrink
            else:
                margins["left"][key] = 0

            if c < phase_cols - 1:
                margins["right"][key] = x_shrink
                right_key = f"{r},{c+1}"
                margins["left"][right_key] = x_shrink
            else:
                margins["right"][key] = 0
    return margins     
            
def get_neighbor_phase_index(y, x, direction, tiles_per_phase_y, tiles_per_phase_x, phase_cols, phase_rows):
    row = y // tiles_per_phase_y
    col = x // tiles_per_phase_x

    if direction == "top" and row > 0:
        row -= 1
    elif direction == "bottom" and row < phase_rows - 1:
        row += 1
    elif direction == "left" and col > 0:
        col -= 1
    elif direction == "right" and col < phase_cols - 1:
        col += 1
    else:
        return None

    if 0 <= row < phase_rows and 0 <= col < phase_cols:
        return row * phase_cols + col
    return None

def get_current_phase_index(y, x, tiles_per_phase_y, tiles_per_phase_x, phase_cols, phase_rows):
    row = y // tiles_per_phase_y
    col = x // tiles_per_phase_x

    if 0 <= row < phase_rows and 0 <= col < phase_cols:
        return row * phase_cols + col
    return None

def generate_blend_mask(grid_height, grid_width, phase_rows, phase_cols,
                        tiles_per_phase_y, tiles_per_phase_x, transition_layer):
    blend_mask = np.zeros((grid_height, grid_width), dtype=bool)
    direction_map = [[set() for _ in range(grid_width)] for _ in range(grid_height)]
    blended_phases_map = [[set() for _ in range(grid_width)] for _ in range(grid_height)]
    for pr in range(phase_rows):
        for pc in range(phase_cols):
            key = f"{pr},{pc}"
            top = transition_layer["top"].get(key, 0)
            bottom = transition_layer["bottom"].get(key, 0)
            left = transition_layer["left"].get(key, 0)
            right = transition_layer["right"].get(key, 0)
            #own_phase_idx = pr * phase_cols + pc
            start_y = pr * tiles_per_phase_y
            start_x = pc * tiles_per_phase_x
            # TOP
            for dy in range(1, top + 1):
                y = start_y - dy
                if 0 <= y < grid_height:
                    for x in range(start_x, start_x + tiles_per_phase_x):
                        if 0 <= x < grid_width:
                            blend_mask[y, x] = True
                            direction_map[y][x].add("bottom")
                            #blended_phases_map[y][x].add(own_phase_idx)
            # BOTTOM
            for dy in range(1, bottom + 1):
                y = start_y + tiles_per_phase_y - 1 + dy
                if 0 <= y < grid_height:
                    for x in range(start_x, start_x + tiles_per_phase_x):
                        if 0 <= x < grid_width:
                            blend_mask[y, x] = True
                            direction_map[y][x].add("top")
                            #reversed
            # LEFT
            for dx in range(1, left + 1):
                x = start_x - dx
                if 0 <= x < grid_width:
                    for y in range(start_y, start_y + tiles_per_phase_y):
                        if 0 <= y < grid_height:
                            blend_mask[y, x] = True
                            direction_map[y][x].add("right")
            # RIGHT
            for dx in range(1, right + 1):
                x = start_x + tiles_per_phase_x - 1 + dx
                if 0 <= x < grid_width:
                    for y in range(start_y, start_y + tiles_per_phase_y):
                        if 0 <= y < grid_height:
                            blend_mask[y, x] = True
                            direction_map[y][x].add("left")
    for pr in range(phase_rows):
        for pc in range(phase_cols):
            key = f"{pr},{pc}"
            top = transition_layer["top"].get(key, 0)
            bottom = transition_layer["bottom"].get(key, 0)
            left = transition_layer["left"].get(key, 0)
            right = transition_layer["right"].get(key, 0)                       
            start_y = pr * tiles_per_phase_y
            start_x = pc * tiles_per_phase_x
            y_end = min(start_y + tiles_per_phase_y, grid_height)
            x_end = min(start_x + tiles_per_phase_x, grid_width)
            #own
            for y in range(start_y, y_end):
                for x in range(start_x, x_end):
                    if blend_mask[y, x]:
                        own_idx = get_current_phase_index(y, x, tiles_per_phase_y, tiles_per_phase_x, phase_cols, phase_rows)
                        if own_idx is not None:
                            blended_phases_map[y][x].add(own_idx)
            # TOP
            for dy in range(1, top + 1):
                y = start_y - dy
                if 0 <= y < grid_height:
                    for x in range(start_x, start_x + tiles_per_phase_x):
                        if 0 <= x < grid_width:
                            neighbor_idx = get_neighbor_phase_index(y, x, "bottom",
                                tiles_per_phase_y, tiles_per_phase_x, phase_cols, phase_rows)
                            if neighbor_idx is not None:
                                blended_phases_map[y][x].add(neighbor_idx)
                            else:
                                print(f"[Warning] No neighbor phase for ({y}, {x}) in top direction")
            # # BOTTOM
            for dy in range(1, bottom + 1):
                y = start_y + tiles_per_phase_y - 1 + dy
                if 0 <= y < grid_height:
                    for x in range(start_x, start_x + tiles_per_phase_x):
                        if 0 <= x < grid_width:
                            neighbor_idx = get_neighbor_phase_index(y, x, "top",
                                tiles_per_phase_y, tiles_per_phase_x, phase_cols, phase_rows)
                            if neighbor_idx is not None:
                                blended_phases_map[y][x].add(neighbor_idx)
                            else:
                                print(f"[Warning] No neighbor phase for ({y}, {x}) in BOTTOM direction")
            #  LEFT
            for dx in range(1, left + 1):
                x = start_x - dx
                if 0 <= x < grid_width:
                    for y in range(start_y, start_y + tiles_per_phase_y):
                        if 0 <= y < grid_height:
                            neighbor_idx = get_neighbor_phase_index(y, x, "right",
                                tiles_per_phase_y, tiles_per_phase_x, phase_cols, phase_rows)
                            if neighbor_idx is not None:
                                blended_phases_map[y][x].add(neighbor_idx)
            #  RIGHT
            for dx in range(1, right + 1):
                x = start_x + tiles_per_phase_x - 1 + dx
                if 0 <= x < grid_width:
                    for y in range(start_y, start_y + tiles_per_phase_y):
                        if 0 <= y < grid_height:
                            neighbor_idx = get_neighbor_phase_index(y, x, "left",
                                tiles_per_phase_y, tiles_per_phase_x, phase_cols, phase_rows)
                            if neighbor_idx is not None:
                                blended_phases_map[y][x].add(neighbor_idx)
    return blend_mask, direction_map, blended_phases_map


def get_phase_idx_at(x, y, phase_map):
    return phase_map[y][x] 
   
def get_effective_tile_policy(x, y, phases, phase_map, frequency_matrix, mode, weight, blend_mask):
    phase_idx = get_phase_idx_at(x, y, phase_map)
    phase = phases[phase_idx]

    freqs = phase["tile_frequencies"]
    allowed = phase["allowed_tiles"]
    
    # Construct possibility matrix from allowed
    possibility_matrix = np.zeros((4, 1), dtype=int)
    for t in [0, 2, 3, 4]:
        if t in allowed:
            possibility_matrix[[0, 2, 3, 4].index(t), 0] = 1

    placed_counts = {}
    total_placed = 0
    for yy in range(y):
        for xx in range(frequency_matrix.shape[1]):
            if (
                phase_map[yy][xx] == phase_idx and not blend_mask[yy, xx] # <-- new line
            ):
                val = frequency_matrix[yy, xx]
                placed_counts[val] = placed_counts.get(val, 0) + 1
                total_placed += 1

    adjusted_probs = adjust_probabilities(
        frequency_targets=freqs,
        placed_counts=placed_counts,
        total_tiles_placed=total_placed,
        tile_possibility_matrix=possibility_matrix,
        mode=mode,
        weight=weight
    )
    return adjusted_probs, possibility_matrix

def get_effective_transition_tile_policy(
    x, y,
    phase_map, blended_phases_map, phases,
    constraints, direction_map,
    frequency_matrix=None,
    blend_mask=None
):
    VALID_TILES = {0, 2, 3, 4}
    current_phase = phase_map[y][x]

    phase_indices_raw = blended_phases_map[y][x]
    if isinstance(phase_indices_raw, str):
        phase_indices = [int(p) for p in phase_indices_raw.split(";")]
    elif isinstance(phase_indices_raw, (set, list, tuple)):
        phase_indices = list(phase_indices_raw)
    else:
        phase_indices = []

    val = direction_map[y][x]
    directions = val if isinstance(val, (set, list)) else {val} if pd.notna(val) else set()
    direction = next(iter(directions)) if directions else None  

    def clean_tile_frequencies(freqs, valid):
        return {int(k): float(v) for k, v in freqs.items() if int(k) in valid}

    freqs_by_phase = {
        pid: clean_tile_frequencies(phases[pid].get("tile_frequencies", {}), VALID_TILES)
        for pid in phase_indices
    }

    region_tag = blended_phases_map[y][x]
    H = W = 0
    if frequency_matrix is not None:
        H, W = frequency_matrix.shape

    def get_subregion_bounds(region_tag):
        region_coords = [(yy, xx) for yy in range(H) for xx in range(W)
                        if blended_phases_map[yy][xx] == region_tag]
        if not region_coords:
            return y, y, x, x

        ys = [yy for yy, _ in region_coords]
        xs = [xx for _, xx in region_coords]
        return min(ys), max(ys), min(xs), max(xs)

    region_min_y, region_max_y, region_min_x, region_max_x = get_subregion_bounds(region_tag)

    def compute_alpha_from_box(min_y, max_y, min_x, max_x, direction):
        if direction == "top":
            return (max_y - y) / max((max_y - min_y), 1)  
        elif direction == "bottom":
            return (y - min_y) / max((max_y - min_y), 1)  
        elif direction == "left":
            return (max_x - x) / max((max_x - min_x), 1)  # reversed
        elif direction == "right":
            return (x - min_x) / max((max_x - min_x), 1)
        else:
            norm_y = (y - min_y) / max((max_y - min_y), 1)
            norm_x = (x - min_x) / max((max_x - min_x), 1)
            return 0.5 * (norm_x + norm_y)

    alpha = compute_alpha_from_box(region_min_y, region_max_y, region_min_x, region_max_x, direction)

    if len(phase_indices) == 2:
        pid_from = current_phase
        pid_to = [pid for pid in phase_indices if pid != current_phase][0]
        if direction in {"bottom", "right"}:
            weights = {
                pid_from: 1.0 - alpha,
                pid_to: alpha
            }
        elif direction in {"top", "left"}:
            weights = {
                pid_to: alpha,
                pid_from: 1.0 - alpha
            }
        else:
            weights = {pid: 0.5 for pid in phase_indices}
    elif len(phase_indices) == 3:
        # Separate main and neighbor phases
        pid_main = current_phase
        pid_neighbors = [pid for pid in phase_indices if pid != pid_main]

        # Initialize direction-based assignment
        pid_top_bottom = None
        pid_left_right = None

        # Assign pids based on known direction labels
        if "top" in directions or "bottom" in directions:
            pid_top_bottom = pid_neighbors[0]
            pid_left_right = pid_neighbors[1]
        elif "left" in directions or "right" in directions:
            pid_left_right = pid_neighbors[0]
            pid_top_bottom = pid_neighbors[1]

        # Assign alpha and beta weights (row and column progress)
        beta = (x - region_min_x) / max((region_max_x - region_min_x), 1)
        alpha = (y - region_min_y) / max((region_max_y - region_min_y), 1)

        # Scale weights — adjust coefficients if needed
        main_weight = max(0.0, 1.0 - alpha * 0.4 - beta * 0.4)
        weights = {
            pid_main: main_weight,
            pid_top_bottom: alpha * 0.4,
            pid_left_right: beta * 0.4,
        }

    # Blend tile frequencies
    blended_freq = {t: 0.0 for t in VALID_TILES}
    for pid, weight in weights.items():
        for t in VALID_TILES:
            val = freqs_by_phase[pid].get(t, 0.0)
            blended_freq[t] += weight * val

    # Step 2: Normalize the whole distribution
    total = sum(blended_freq.values())
    if total > 0:
        blended_freq = {t: v / total for t, v in blended_freq.items()}
    else:
        blended_freq = {t: 1.0 / len(VALID_TILES) for t in VALID_TILES}

    # Rectangular Memory Region Correction
    actual_counts = {}
    if frequency_matrix is not None and region_tag:
        for yy in range(region_min_y, y + 1):
            for xx in range(region_min_x, region_max_x + 1):
                if (yy < y or (yy == y and xx < x)) and blended_phases_map[yy][xx] == region_tag:
                    t = frequency_matrix[yy][xx]
                    actual_counts[t] = actual_counts.get(t, 0) + 1
        total_placed = sum(actual_counts.values())
    else:
        total_placed = 0

    # Apply constraints
    allowed_tiles = set()
    for pid in phase_indices:
        allowed_tiles.update(phases[pid].get("allowed_tiles", []))
    constraint_allowed = set()
    if isinstance(constraints, dict):
        for k in ["disallow_tile_types", "allowed_tiles"]:
            constraint_allowed.update({t for t in constraints.get(k, []) if t in VALID_TILES})
    if constraint_allowed:
        allowed_tiles &= constraint_allowed

    # Possibility matrix
    possibility_matrix = np.zeros((4, 1), dtype=int)
    for t in allowed_tiles:
        if t in VALID_TILES:
            possibility_matrix[[0, 2, 3, 4].index(t), 0] = 1

    adjusted_probs = adjust_probabilities(
        frequency_targets=blended_freq,
        placed_counts=actual_counts,
        total_tiles_placed=total_placed,
        tile_possibility_matrix=possibility_matrix,
        mode="transition",
        weight=alpha
    )
    # with open(LOG_FILE, mode='a', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow([
    #         x, y,
    #         direction,
    #         round(alpha, 3),
    #         ";".join(map(str, phase_indices)),
    #         current_phase,
    #         str(weights),
    #         str(blended_freq),
    #         str(actual_counts),
    #         str(adjusted_probs)
    #     ])
    return adjusted_probs, possibility_matrix

def clear_output_folder(folder_path, exceptions=None):
    if exceptions is None:
        exceptions = []
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            if filename in exceptions:
                continue  
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    else:
        print(f"The folder {folder_path} does not exist.")
#######################################################################################################################################################
#PyQt5 GUI - user input
class PhaseTab(QWidget):
    def __init__(self, phase_name):
        super().__init__()
        self.advanced_rules = {
            "Rule 1:❌ Disallowed Propagation": "disallow_tile_3_noT_after_tile_3_noT",
            "Rule 2:❌ Disallowed Propagation": "disallow_tile_3_noB_after_tile_3_noB",
            "Rule 3:❌ Disallowed Propagation": "disallow_tile_3_noL_after_tile_3_noR",
            "Rule 4:❌ Disallowed Propagation": "disallow_tile_3_noR_after_tile_3_noL",
            "Rule 5:❌ Disallowed Propagation": "disallow_tile_4_after_tile_4",
            "Rule 6:❌ Disallowed Propagation": "disallow_tile_0_after_tile_0",
            "Rule 7:❌ Disallowed Propagation": "disallow_tile_2_RT_after_tile_2_LB",
            "Rule 8:❌ Disallowed Propagation": "disallow_tile_2_LR_after_tile_2_LR"
        }
        self.advanced_checkboxes = {}
        self.phase_name = phase_name
        self.layout = QVBoxLayout()
        # Coordination input
        self.coord_input = QDoubleSpinBox()
        self.coord_input.setRange(0, 4)
        self.coord_input.setSingleStep(0.1)
        self.coord_input.setValue(2.0)
        self.layout.addWidget(QLabel(f"Coordination Number for {phase_name}"))
        self.layout.addWidget(self.coord_input)

        # Manual input checkbox
        self.manual_input = QCheckBox("Input Own frequencies")
        self.manual_input.stateChanged.connect(self.toggle_frequency_spinbox)
        self.layout.addWidget(self.manual_input)

        # Generate button
        self.generate_button = QPushButton("Generate Matching frequencies")
        self.generate_button.clicked.connect(self.generate_random_frequencies)
        self.layout.addWidget(self.generate_button)

        # Feedback and current coord
        self.feedback_label = QLabel()
        self.current_coord_label = QLabel("Current Coordination: 0.00")
        self.layout.addWidget(self.current_coord_label)
        self.layout.addWidget(self.feedback_label)

        # Tile sum controls
        self.tile_checkboxes = {}
        self.frequency_spinbox = {}
        self.frequency_labels = {}

        tile_box = QGroupBox("Tile Type frequency")
        tile_layout = QFormLayout()
        tile_box.setLayout(tile_layout)
        self.layout.addWidget(tile_box)
        for t in [0, 2, 3, 4]:
            cb = QCheckBox(f"Allow coordination {t}")
            spinbox = QSpinBox()
            spinbox.setRange(0, 100)
            spinbox.setSingleStep(1)
            spinbox.setValue(0)
            spinbox.setEnabled(False)

            percent_label = QLabel("0%")
            self.frequency_labels[t] = percent_label

            tile_name = f"Tile R type {t}"
            example_tile = self.get_example_tile_for_type(t)
            preview = self.create_tile_preview(example_tile)

            row = QWidget()
            row_layout = QHBoxLayout()
            row_layout.addWidget(QLabel(tile_name))
            row_layout.addWidget(preview)
            row_layout.addWidget(cb)
            row_layout.addWidget(spinbox)
            row_layout.addWidget(percent_label)
            row.setLayout(row_layout)

            tile_layout.addRow(row)

            # Signal connections
            spinbox.valueChanged.connect(lambda val, r=t: self.update_spinbox_label(r))
            spinbox.valueChanged.connect(self.validate_frequency_match)

            def make_toggle_handler(spinbox=spinbox, label=percent_label, t=t):
                def toggle(state):
                    enabled = state == Qt.Checked
                    spinbox.setEnabled(enabled and self.manual_input.isChecked())
                    if not enabled:
                        spinbox.setValue(0)
                        label.setText("0%")
                    self.validate_frequency_match()
                return toggle

            cb.stateChanged.connect(make_toggle_handler())

            self.tile_checkboxes[t] = cb
            self.frequency_spinbox[t] = spinbox
            
            # row = QWidget()
            # row_layout = QHBoxLayout()
            # row_layout.addWidget(cb)
            # row_layout.addWidget(spinbox)
            # row_layout.addWidget(percent_label)
            # row.setLayout(row_layout)
            # tile_layout.addRow(f"Tile R type {t}", row) 
        # --- Group for Advanced Connectivity Rules ---
        self.advanced_group = QGroupBox(f"Advanced Connectivity Rules for {phase_name}")
        advanced_group_layout = QVBoxLayout()

        # Checkbox to enable advanced rule assignment
        self.advanced_toggle = QCheckBox("Assign Advanced Connectivity Rules")
        self.advanced_toggle.stateChanged.connect(self.toggle_advanced_rules)
        advanced_group_layout.addWidget(self.advanced_toggle)

        # Inner GroupBox for the actual rule checkboxes
        self.advanced_rules_box = QGroupBox("Ban Propagation Patterns")
        self.advanced_rules_layout = QVBoxLayout()
        self.advanced_rules_box.setLayout(self.advanced_rules_layout)

        for label, key in self.advanced_rules.items():
            cb = QCheckBox(label)
            cb.setEnabled(False)

            tile_after = key.split("_after_")[1]
            tile_before = key.replace("disallow_", "").split("_after_")[0]
            viz = self.create_rule_preview(tile_before, tile_after)

            row = QWidget()
            row_layout = QHBoxLayout()
            row_layout.addWidget(cb)
            row_layout.addWidget(viz)
            row.setLayout(row_layout)
            
            self.advanced_rules_layout.addWidget(row)
            self.advanced_checkboxes[key] = cb
        self.advanced_rules_box.setVisible(False)
        advanced_group_layout.addWidget(self.advanced_rules_box)
        self.advanced_group.setLayout(advanced_group_layout)
        self.layout.addWidget(self.advanced_group)
        self.setLayout(self.layout)   
    def update_spinbox_label(self, r):
        val = self.frequency_spinbox[r].value()
        self.frequency_labels[r].setText(f"{val}%")
        
    def get_example_tile_for_type(self, t):
        if t == 0:
            return "tile_0"
        elif t == 2:
            return "tile_2_LR"
        elif t == 3:
            return "tile_3_noT"  
        elif t == 4:
            return "tile_4"  
        else:
            return []
    
    def toggle_frequency_spinbox(self):
        enabled = self.manual_input.isChecked()
        for r in self.frequency_spinbox:
            self.frequency_spinbox[r].setEnabled(enabled and self.tile_checkboxes[r].isChecked())
        self.validate_frequency_match()
        
    def generate_random_frequencies(self):
        allowed = [r for r in self.tile_checkboxes if self.tile_checkboxes[r].isChecked()]
        if not allowed:
            self.feedback_label.setText("❌ No types selected.")
            return
        target = self.coord_input.value()

        for _ in range(10000):
            weights = [random.random() for _ in allowed]
            normed = [w / sum(weights) for w in weights]
            check = sum(r * d for r, d in zip(allowed, normed))
            if abs(check - target) < 0.01:
                # Round and correct total to 100%
                raw_percents = [round(d * 100) for d in normed]
                diff = 100 - sum(raw_percents)

                # Fix imbalance by adjusting values up/down
                for _ in range(abs(diff)):
                    if diff > 0:
                        idx = raw_percents.index(min(raw_percents))
                        raw_percents[idx] += 1
                    elif diff < 0:
                        idx = raw_percents.index(max(raw_percents))
                        raw_percents[idx] -= 1
                for i, r in enumerate(allowed):
                    self.frequency_spinbox[r].setValue(raw_percents[i])
                self.validate_frequency_match()
                self.feedback_label.setText("✅ Frequencies match coordination number.")
                return
        self.feedback_label.setText("❌ Couldn't match coordination target. Try other tile types.")

    def validate_frequency_match(self):
        total_frequency = 0
        weighted_sum = 0
        for t in self.tile_checkboxes:
            if self.tile_checkboxes[t].isChecked():
                val = self.frequency_spinbox[t].value() / 100.0
                weighted_sum += val * t
                total_frequency += val
        self.current_coord_label.setText(f"Current Coordination: {weighted_sum:.2f}")
        if not self.manual_input.isChecked():
            return
        if abs(total_frequency - 1.0) > 0.01:
            self.feedback_label.setText("❌ Frequencies must sum to 1.0")
            return
        target = self.coord_input.value()
        if abs(weighted_sum - target) > 0.02:
            self.feedback_label.setText(f"❌ Coordination mismatch (target: {target:.2f})")
        else:
            self.feedback_label.setText("✅ Coordination matches within 0.02 accuracy ")
            
    def toggle_advanced_rules(self, state):
        is_checked = state == Qt.Checked
        self.advanced_rules_box.setVisible(is_checked)
        for cb in self.advanced_checkboxes.values():
            cb.setEnabled(is_checked)
            
    def create_tile_preview(self, tile_type, tile_size=1.0):
        dwg = svgwrite.Drawing(size=(tile_size, tile_size))

        group = render_tile_svg_group(
            create_tile_geometry(tile_type, tile_size),
            tile_size,
            flip_x=False,
            invert_y=True
        )

        dwg.add(group)

        svg_str = dwg.tostring()
        byte_data = QByteArray(svg_str.encode("utf-8"))
        svg_widget = QSvgWidget()
        svg_widget.load(byte_data)
        svg_widget.setFixedSize(40, 40)
        return svg_widget
            
    def create_rule_preview(self, tile_before, tile_after, tile_size=1.0):
        dwg = svgwrite.Drawing(size=(tile_size * 2, tile_size))

        # Create visual groups
        group1 = render_tile_svg_group(create_tile_geometry(tile_before, tile_size), tile_size, flip_x=False, invert_y=True)
        group2 = render_tile_svg_group(create_tile_geometry(tile_after, tile_size), tile_size, flip_x=False, invert_y=True)

        # Move AFTER tile to the RIGHT instead
        group1.translate(tile_size, 0)  # move AFTER tile to right

        # Draw BEFORE tile first, then AFTER tile
        dwg.add(group2)  # after
        dwg.add(group1)  # before
        

        svg_str = dwg.tostring()
        byte_data = QByteArray(svg_str.encode("utf-8"))
        svg_widget = QSvgWidget()
        svg_widget.load(byte_data)
        svg_widget.setFixedSize(80, 40)
        return svg_widget
    
class MainApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("User Inputs")
        self.main_layout = QVBoxLayout(self)
        # Stacked widget to hold pages
        self.stacked_widget = QStackedWidget()
        self.main_layout.addWidget(self.stacked_widget)
        # Page 1
        self.page1 = QWidget()
        self.page1_layout = QVBoxLayout()
        self.page1.setLayout(self.page1_layout)
        #Page 2 content
        self.page2_content = QWidget()
        self.page2_layout = QVBoxLayout()
        self.page2_content.setLayout(self.page2_layout)
        # Scroll area for page 2
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.page2_content)
        # Add pages to stacked widget
        self.stacked_widget.addWidget(self.page1)         
        self.stacked_widget.addWidget(scroll_area)        
        # load settings
        load_btn = QPushButton("Load Settings File")
        load_btn.clicked.connect(self.load_settings_from_file)
        self.page1_layout.addWidget(load_btn)
        # Grid size
        size_layout = QHBoxLayout()
        sample_group = QGroupBox("Sample size determination")
        sample_group_layout = QVBoxLayout()
        self.stl_width_mm = QSpinBox()
        self.stl_height_mm = QSpinBox()
        self.stl_width_mm.setRange(10, 100)
        self.stl_height_mm.setRange(10,100)
        self.stl_width_mm.setValue(25)
        self.stl_height_mm.setValue(25)
        size_layout.addWidget(QLabel("Sample Width (mm):"))
        size_layout.addWidget(self.stl_width_mm)
        size_layout.addWidget(QLabel("Sample Length (mm):"))
        size_layout.addWidget(self.stl_height_mm)
        sample_group_layout.addLayout(size_layout)   
        sample_group.setLayout(sample_group_layout)
        self.page1_layout.addWidget(sample_group) 
        self.stl_height_mm.valueChanged.connect(self.update_phase_config)
        self.stl_width_mm.valueChanged.connect(self.update_phase_config)
        # Blending controls
        self.blend_checkbox = QCheckBox("Enable Transition Layer")
        self.harshness_input = QDoubleSpinBox()
        self.harshness_input.setRange(0.0, 0.4)
        self.harshness_input.setSingleStep(0.05)
        self.harshness_input.setValue(0.1)
        blend_layout = QHBoxLayout()
        blend_layout.addWidget(self.blend_checkbox)
        blend_layout.addWidget(QLabel("Transition size (% of Phase):"))
        blend_layout.addWidget(self.harshness_input)
        transition_group = QGroupBox("Transition Layer Options")
        transition_group.setLayout(blend_layout)
        self.page1_layout.addWidget(transition_group)
        self.blend_checkbox.stateChanged.connect(self.update_phase_config)
        self.harshness_input.valueChanged.connect(self.update_phase_config)
        # STL Parameters Group
        stl_group = QGroupBox("STL Output Settings")
        stl_group_layout = QVBoxLayout()
        # STL choice checkbox
        stl_top_row_layout = QHBoxLayout()
        self.stl_choice_checkbox = QCheckBox("Generate STL files")
        self.stl_choice_checkbox.setChecked(True)
        stl_top_row_layout.addWidget(self.stl_choice_checkbox)
        stl_top_row_layout.addSpacing(20)
        stl_top_row_layout.addWidget(QLabel("Number of Saved Samples:"))
        self.num_samples_spinbox = QSpinBox()
        self.num_samples_spinbox.setRange(1, 3)
        self.num_samples_spinbox.setValue(1)
        stl_top_row_layout.addWidget(self.num_samples_spinbox)
        stl_top_row_layout.addWidget(QLabel("Iterations over Simulation (packages of 5):"))
        self.iterations_spinbox = QSpinBox()
        self.iterations_spinbox.setRange(1, 25)  
        self.iterations_spinbox.setSingleStep(1)  
        self.iterations_spinbox.setValue(10)      
        self.iterations_spinbox.setToolTip("Each iteration runs 5 independent simulations.")
        stl_top_row_layout.addWidget(self.iterations_spinbox)
        stl_group_layout.addLayout(stl_top_row_layout)
        # Tile size input 
        tile_size_layout = QHBoxLayout()
        self.tile_size_mm = QDoubleSpinBox()
        self.tile_size_mm.setRange(0.5, 3.0)
        self.tile_size_mm.setSingleStep(0.05)
        self.tile_size_mm.setValue(0.5)
        tile_size_layout.addWidget(QLabel("Tile Size [mm]:"))
        tile_size_layout.addWidget(self.tile_size_mm)
        stl_group_layout.addLayout(tile_size_layout)
        self.tile_size_mm.valueChanged.connect(self.update_thickness_constraints)
        self.tile_size_mm.valueChanged.connect(self.update_phase_config)
        # Line thickness input
        thickness_layout = QHBoxLayout()
        self.line_thickness = QDoubleSpinBox()
        self.line_thickness.setSingleStep(0.05)
        self.line_thickness.setValue(0.1)
        self.update_thickness_constraints()
        thickness_layout.addWidget(QLabel("Inclusion strut Thickness [mm]:"))
        thickness_layout.addWidget(self.line_thickness)
        stl_group_layout.addLayout(thickness_layout)
        # Extrusion height input
        extrusion_layout = QHBoxLayout()
        self.extrusion_height = QDoubleSpinBox()
        self.extrusion_height.setRange(1.0, 10.0)
        self.extrusion_height.setSingleStep(0.05)
        self.extrusion_height.setValue(1.0)
        extrusion_layout.addWidget(QLabel("Extrusion Height [mm]:"))
        extrusion_layout.addWidget(self.extrusion_height)
        stl_group_layout.addLayout(extrusion_layout)
        # Outer walls checkboxes
        grid_wall_group = QGroupBox("Add Outer Walls")
        grid_wall_layout = QHBoxLayout()
        self.add_horizontal_walls = QCheckBox("Top & Bottom")
        self.add_vertical_walls = QCheckBox("Left & Right")
        self.add_horizontal_walls.setChecked(True)
        self.add_vertical_walls.setChecked(True)
        grid_wall_layout.addWidget(self.add_horizontal_walls)
        grid_wall_layout.addWidget(self.add_vertical_walls)
        grid_wall_group.setLayout(grid_wall_layout)
        stl_group_layout.addWidget(grid_wall_group)
        self.add_horizontal_walls.stateChanged.connect(self.update_phase_config)
        self.add_vertical_walls.stateChanged.connect(self.update_phase_config)
        stl_group.setLayout(stl_group_layout)
        self.page1_layout.addWidget(stl_group)
        # Phase layout sliders and grid preview
        self.phase_rows_slider = QSlider(Qt.Vertical)
        self.phase_rows_slider.setRange(1, 4)
        self.phase_rows_slider.setValue(2)
        self.phase_rows_slider.setFixedHeight(100)
        self.phase_cols_slider = QSlider(Qt.Horizontal)
        self.phase_cols_slider.setRange(1, 4)
        self.phase_cols_slider.setValue(2)
        self.phase_cols_slider.setFixedWidth(100)
        # Phase preview 4x4 grid
        self.phase_preview_grid = QGridLayout()
        self.phase_grid_cells = []
        for r in range(4):
            row = []
            for c in range(4):
                cell = QLabel()
                cell.setFixedSize(40, 40)
                cell.setStyleSheet("background-color: lightgray; border: 1px solid black;")
                cell.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self.phase_preview_grid.addWidget(cell, r, c)
                row.append(cell)
            self.phase_grid_cells.append(row)
        # Wrap the grid in a container for layouting
        grid_preview_container = QWidget()
        grid_preview_container.setLayout(self.phase_preview_grid)
        # Style the sliders
        slider_style = """
        QSlider::groove:horizontal { height: 6px; }
        QSlider::handle:horizontal { background: gray; width: 12px; margin: -4px 0; }
        QSlider::groove:vertical { width: 6px; }
        QSlider::handle:vertical { background: gray; height: 12px; margin: 0 -4px; }
        """
        self.phase_rows_slider.setStyleSheet(slider_style)
        self.phase_cols_slider.setStyleSheet(slider_style)
        # Live value labels
        self.rows_label = QLabel("2")
        self.cols_label = QLabel("2")
        # Connect slider updates
        self.phase_rows_slider.valueChanged.connect(lambda val: self.rows_label.setText(str(val)))
        self.phase_cols_slider.valueChanged.connect(lambda val: self.cols_label.setText(str(val)))
        self.phase_rows_slider.valueChanged.connect(self.update_phase_config)
        self.phase_cols_slider.valueChanged.connect(self.update_phase_config)
        self.phase_rows_slider.valueChanged.connect(self.update_phase_grid_preview)
        self.phase_cols_slider.valueChanged.connect(self.update_phase_grid_preview)
        # Layout sliders and preview
        slider_layout = QHBoxLayout()
        # Row slider
        row_slider_layout = QVBoxLayout()
        row_slider_layout.addWidget(QLabel("Rows"))
        row_slider_layout.addWidget(self.phase_rows_slider)
        row_slider_layout.addWidget(self.rows_label)
        # Column slider
        col_slider_layout = QVBoxLayout()
        col_slider_layout.addWidget(QLabel("Columns"))
        col_slider_layout.addWidget(self.cols_label)
        col_slider_layout.addWidget(self.phase_cols_slider)
        # Final layout: Row slider | Grid preview | Col slider
        slider_layout.addLayout(row_slider_layout)
        slider_layout.addWidget(grid_preview_container)
        slider_layout.addLayout(col_slider_layout)
        phase_layout_group = QGroupBox("Grid Division into Phases")
        phase_layout_group.setLayout(slider_layout)
        self.page1_layout.addWidget(phase_layout_group)
        # Output folder path
        output_path_layout = QHBoxLayout()
        self.output_path = QLineEdit()
        browse_button = QPushButton("Browse Output Folder")
        browse_button.clicked.connect(self.select_output_folder)
        output_path_layout.addWidget(QLabel("Output Folder:"))
        output_path_layout.addWidget(self.output_path)
        output_path_layout.addWidget(browse_button)
        self.page1_layout.addLayout(output_path_layout)
        # Buttons
        self.grid_info_label = QLabel("")
        self.grid_info_label.setTextFormat(Qt.RichText)
        self.page1_layout.addWidget(self.grid_info_label)
        self.next_button = QPushButton("Next: Configure Phases")
        self.next_button.clicked.connect(self.go_to_step_2)
        self.page1_layout.addWidget(self.next_button)

        self.phase_tab_highlight_grid = QGridLayout()
        self.phase_tab_cells = []
        for r in range(4):
            row = []
            for c in range(4):
                cell = QLabel()
                cell.setFixedSize(40, 40)
                cell.setStyleSheet("background-color: lightgray; border: 1px solid black;")
                cell.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self.phase_tab_highlight_grid.addWidget(cell, r, c)
                row.append(cell)
            self.phase_tab_cells.append(row)

        grid_wrapper = QWidget()
        grid_wrapper.setLayout(self.phase_tab_highlight_grid)
        grid_wrapper.setFixedHeight(200)

        # --- Phase config tabs --
        self.tabs = QTabWidget()
        self.tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        # --- Group container with title ---
        phase_config_group = QGroupBox("Phase-Specific Configuration")
        phase_config_layout = QVBoxLayout()
        phase_config_layout.addWidget(grid_wrapper)
        phase_config_layout.addWidget(self.tabs)
        phase_config_group.setLayout(phase_config_layout)

        # --- Scroll container for entire content ---
        tab_scroll_area = QScrollArea()
        tab_scroll_area.setWidgetResizable(True)
        tab_scroll_area.setWidget(phase_config_group)

        self.page2_layout.addWidget(tab_scroll_area)

        # Navigation buttons
        self.back_button = QPushButton("⬅ Back to Settings")
        self.back_button.clicked.connect(self.go_to_step_1)
        self.page2_layout.addWidget(self.back_button)

        self.run_button = QPushButton("Save and Close")
        self.run_button.clicked.connect(self.collect_and_close)
        self.page2_layout.addWidget(self.run_button)

        # Setup complete
        self.update_phase_config()
        self.tabs.currentChanged.connect(self.highlight_current_phase)
        
    def go_to_step_1(self):
        self.stacked_widget.setCurrentIndex(0)   
    def go_to_step_2(self):
        self.stacked_widget.setCurrentIndex(1) 

    def update_thickness_constraints(self):
        tile_size = self.tile_size_mm.value()
        min_thick = min(0.1, tile_size/5)
        max_thick = tile_size/3
        self.line_thickness.setRange(min_thick, max_thick)
        current_val = self.line_thickness.value()
        if current_val < min_thick:
            self.line_thickness.setValue(min_thick)
        elif current_val > max_thick:
            self.line_thickness.setValue(max_thick)
        
    def load_settings_from_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select settings JSON file", "", "JSON Files (*.json)"
        )
        if not file_path:
            return
        if not os.path.exists(file_path):
            QMessageBox.warning(self, "File Not Found", "No input_settings.json found in the selected folder.")
            return
        with open(file_path, "r") as f:
            loaded = json.load(f)
        if "stl_dimensions_mm" in loaded:
            desired_dims = loaded["stl_dimensions_mm"].get("desired", [50, 80])
            self.stl_width_mm.setValue(int(desired_dims[0]))
            self.stl_height_mm.setValue(int(desired_dims[1]))
        layout = loaded.get("phase_layout", {})
        self.phase_rows_slider.setValue(int(layout.get("rows", 2)))
        self.phase_cols_slider.setValue(int(layout.get("cols", 2)))
        self.tile_size_mm.setValue(float(loaded.get("tile_size_mm", 0.2)))
        self.line_thickness.setValue(float(loaded.get("line_thickness", 0.2)))
        self.extrusion_height.setValue(float(loaded.get("extrusion_height", 2.0)))
        self.num_samples_spinbox.setValue(int(loaded.get("num_samples", 1)))
        self.iterations_spinbox.setValue(int(loaded.get("num_iterations", 10)))
        self.stl_choice_checkbox.setChecked(bool(loaded.get("stl", True)))
        walls = loaded.get("add_outer_walls", {})
        self.add_horizontal_walls.setChecked(walls.get("horizontal", True))
        self.add_vertical_walls.setChecked(walls.get("vertical", True))
        self.output_path.setText(str(loaded.get("output_path", "")))
        self.computed_grid_width = loaded.get("grid_width", 0)
        self.computed_grid_height = loaded.get("grid_height", 0)
        self.tiles_per_phase_x = loaded.get("tiles_per_phase_x", 0)
        self.tiles_per_phase_y = loaded.get("tiles_per_phase_y", 0)
        self.blend_checkbox.setChecked(loaded.get("blend_enabled", False))
        self.harshness_input.setValue(loaded.get("blend_harshness", 0.3))
        self.transition_layer = loaded.get("transition_layer", {"x": 0, "y": 0})
        self.update_phase_config()
        phase_map_raw = loaded.get("phase_map")
        if phase_map_raw:
            self.phase_map = np.array(phase_map_raw)
        blend_mask_raw = loaded.get("blend_mask")
        if blend_mask_raw:
            self.blend_mask = np.array(blend_mask_raw)
        blend_dirs_raw = loaded.get("blend_directions")
        if blend_dirs_raw:
            self.blend_directions = [
                [set(d) for d in row] for row in blend_dirs_raw
            ]
        phases = loaded.get("phases", [])
        for i, phase_data in enumerate(phases):
            if i >= len(self.phase_tabs):
                break
            tab = self.phase_tabs[i]
            tab.coord_input.setValue(float(phase_data.get("coordination_target", 2.5)))
            freq = phase_data.get("tile_frequencies", {})
            allowed = phase_data.get("allowed_tiles", [])
            tab.manual_input.setChecked(True)
            for r in [0, 2, 3, 4]:
                tab.tile_checkboxes[r].setChecked(r in allowed)
                tab.frequency_spinbox[r].setEnabled(r in allowed)
                percent = float(freq.get(str(r), 0.0)) * 100
                tab.frequency_spinbox[r].setValue(int(round(percent)))
            connectivity_rules = phase_data.get("connectivity_rules", {})
            for rule_key, checkbox in tab.advanced_checkboxes.items():
                checkbox.setChecked(bool(connectivity_rules.get(rule_key, False)))
            tab.toggle_frequency_spinbox()
            tab.validate_frequency_match()
        print(f"[Loaded settings] from: {file_path}")
        
    def update_phase_grid_preview(self): #page1
        rows = self.phase_rows_slider.value()
        cols = self.phase_cols_slider.value()
        for r in range(4):
            for c in range(4):
                if r < rows and c < cols:
                    self.phase_grid_cells[r][c].setStyleSheet("background-color: lightgreen; border: 1px solid black;")
                    self.phase_grid_cells[r][c].setText(str(r * cols + c + 1))
                else:
                    self.phase_grid_cells[r][c].setStyleSheet("background-color: lightgray; border: 1px solid black;")
                    self.phase_grid_cells[r][c].setText("")
               
    def highlight_current_phase(self, index): #page2
        rows = self.phase_rows_slider.value()
        cols = self.phase_cols_slider.value()

        for r in range(4):
            for c in range(4):
                cell = self.phase_tab_cells[r][c]
                idx = r * cols + c
                if r < rows and c < cols:
                    cell.setText(str(idx + 1))
                    if idx == index:
                        cell.setStyleSheet("background-color: orange; border: 2px solid black;")
                    else:
                        cell.setStyleSheet("background-color: lightgreen; border: 1px solid black;")
                else:
                    cell.setText("")
                    cell.setStyleSheet("background-color: lightgray; border: 1px solid black;")

    def init_phase_tabs(self, count):
        self.tabs.clear()
        self.phase_tabs = []

        for i in range(count):
            tab = PhaseTab(f"Phase {i + 1}")
            self.tabs.addTab(tab, f"Phase {i + 1}")
            self.phase_tabs.append(tab)

    def update_phase_config(self):
        rows = self.phase_rows_slider.value()
        cols = self.phase_cols_slider.value()

        self.init_phase_tabs(rows * cols)
        self.update_phase_grid_preview()
        tile_size_mm = self.tile_size_mm.value()

        # Adjust grid size based on tile size
        raw_grid_w = math.floor(self.stl_width_mm.value() / tile_size_mm)
        raw_grid_h = math.floor(self.stl_height_mm.value() / tile_size_mm)

        wall_h = 2 if self.add_horizontal_walls.isChecked() else 0
        wall_v = 2 if self.add_vertical_walls.isChecked() else 0
        grid_w = raw_grid_w - wall_v
        grid_h = raw_grid_h - wall_h

        # Align to multiples of phase count
        grid_w -= (grid_w-wall_v) % cols
        grid_h -= (grid_h-wall_h) % rows

        self.computed_grid_width = grid_w
        self.computed_grid_height = grid_h

        # Base tiles per phase
        original_tiles_x = grid_w // cols
        original_tiles_y = grid_h // rows

        # transtion layer
        if self.blend_checkbox.isChecked():
            shrink_x = int(original_tiles_x * self.harshness_input.value())
            shrink_y = int(original_tiles_y * self.harshness_input.value())
        else:
            shrink_x = 0
            shrink_y = 0
        self.transition_layer = {"x": shrink_x, "y": shrink_y}

        #Tile count 
        true_tiles_x = self.computed_grid_width
        true_tiles_y = self.computed_grid_height

        # Add wall contribution to final size

        full_grid_w = true_tiles_x + wall_v
        full_grid_h = true_tiles_y + wall_h

        actual_w_mm = full_grid_w * tile_size_mm
        actual_h_mm = full_grid_h * tile_size_mm

        if rows >= 3 and cols >= 3:
            interactions = 4
        elif rows == 1 and cols == 1:
            interactions = 0
        elif rows >= 3 or cols >= 3:
            interactions = 3
        else:
            interactions = 2

        smallest_tiles_x = original_tiles_x
        smallest_tiles_y = original_tiles_y

        if interactions >= 4:
            smallest_tiles_x -= 2*shrink_x
            smallest_tiles_y -= 2*shrink_y
            
        elif interactions == 3:
            smallest_tiles_x -= 2*shrink_x
            smallest_tiles_y -= shrink_y
        elif interactions == 2:
            smallest_tiles_x -= shrink_x
            smallest_tiles_y -= shrink_y
        elif interactions == 1:
            smallest_tiles_x -= shrink_x
            smallest_tiles_y -= shrink_y
        
        self.tiles_per_phase_x=original_tiles_x
        self.tiles_per_phase_y = original_tiles_y        
        wall_msg = f"Outer walls: {'Top/Bottom' if wall_h else ''} {'Left/Right' if wall_v else ''}".strip()

        if shrink_x or shrink_y:
            transition_msg = (
                f"Transition layer: {2*shrink_x}×{2*shrink_y} tiles converted from each interacting phase edges "
                f"to create shared transition zones (Symmetrically subtracted).<br>"
                f"Effective phase area: {smallest_tiles_y}×{smallest_tiles_x} tiles (tansition layer reduction)."
            )
        else:
            transition_msg = "No transition layer."
 
        
        msg = (
            f"<div style='background-color:#eee; padding:8px; border-radius:6px;'>"
            f"Phase layout: {rows}×{cols} phases<br>"
            f"Grid: {true_tiles_x}×{true_tiles_y} tiles → "
            f"{rows}×{cols} phases  of nominal size = {original_tiles_x}×{original_tiles_y} tiles.<br>"
            f"{transition_msg}<br>"
            f"{wall_msg}<br>"
            f"Total physical sample size: {actual_w_mm:.1f} mm × {actual_h_mm:.1f} mm"
            f"</div>"
        )
        self.grid_info_label.setText(msg)

      
    def select_output_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self.output_path.setText(path)     

    def collect_settings(self):
        output_path = self.output_path.text()
        if not output_path:
            QMessageBox.critical(self, "Missing Output Folder", "Select an output folder before continuing.")
            return
        tile_size_mm = float(self.tile_size_mm.value())
        num_samples=  int(self.num_samples_spinbox.value())
        itterations= int(self.iterations_spinbox.value())
        extrusion_height = float(self.extrusion_height.value())
        line_thickness = float(self.line_thickness.value())
        stl_width = int(self.stl_width_mm.value())
        stl_height = int(self.stl_height_mm.value())

        phase_rows = int(self.phase_rows_slider.value())
        phase_cols = int(self.phase_cols_slider.value())
        
        blend_enabled = self.blend_checkbox.isChecked()
        blend_harshness = float(self.harshness_input.value())
        
        phase_map = generate_phase_map(
            grid_width=self.computed_grid_width,
            grid_height=self.computed_grid_height,
            layout_rows=phase_rows,
            layout_cols=phase_cols
        )
        x_shrink = 0
        y_shrink = 0
        if blend_enabled:
            x_shrink = self.transition_layer.get("x", 0)
            y_shrink = self.transition_layer.get("y", 0)

            transition_layer_margins = compute_transition_layer(phase_rows, phase_cols, x_shrink, y_shrink)

            blend_mask, direction_map, blended_phases_map = generate_blend_mask(
                grid_height=self.computed_grid_height,
                grid_width=self.computed_grid_width,
                phase_rows=phase_rows,
                phase_cols=phase_cols,
                tiles_per_phase_y=self.tiles_per_phase_y,
                tiles_per_phase_x=self.tiles_per_phase_x,
                transition_layer=transition_layer_margins
            )

            blended_phases_serialized = [
                [list(cell) for cell in row] for row in blended_phases_map
            ]
        else:
            transition_layer_margins = {
                "top": {}, "bottom": {}, "left": {}, "right": {}
            }
            blend_mask = np.zeros((self.computed_grid_height, self.computed_grid_width), dtype=bool)
            direction_map = [[set() for _ in range(self.computed_grid_width)] for _ in range(self.computed_grid_height)]
            blended_phases_serialized = [
                [[] for _ in range(self.computed_grid_width)] for _ in range(self.computed_grid_height)
            ]

        phases = []
        for i, tab in enumerate(self.phase_tabs):
            coord_target = float(tab.coord_input.value())
            allowed = []
            frequencies = {}

            for r in [0, 2, 3, 4]:
                if tab.tile_checkboxes[r].isChecked():
                    allowed.append(int(r))
                    freq = float(tab.frequency_spinbox[r].value()) / 100.0
                    frequencies[int(r)] = freq

            total = sum(frequencies.values())
            if total > 0:
                frequencies = {int(k): float(v / total) for k, v in frequencies.items()}
            advanced_rules = {
                rule_key: cb.isChecked() for rule_key, cb in tab.advanced_checkboxes.items()
            }
            phases.append({
                "coordination_target": float(coord_target),
                "allowed_tiles": allowed,
                "tile_frequencies": frequencies,
                "blend_enabled": blend_enabled,
                "blend_harshness": blend_harshness,
                "connectivity_rules": advanced_rules,
            })

        return {
            "grid_width": int(self.computed_grid_width),
            "grid_height": int(self.computed_grid_height),
            "output_path": str(output_path),
            "phases": phases,
            "phase_layout": {
                "rows": phase_rows,
                "cols": phase_cols
            },
            "extrusion_height": extrusion_height,
            "line_thickness": line_thickness,
            "tile_size_mm": tile_size_mm,
            "num_samples": num_samples,
            "num_iterations": itterations,
            "stl_dimensions_mm": {
                "desired": [stl_width, stl_height],
                "actual": [
                    float((self.tiles_per_phase_x * phase_cols + 2) * tile_size_mm),
                    float((self.tiles_per_phase_y * phase_rows + 2) * tile_size_mm)
                ]
            },
            "stl": bool(self.stl_choice_checkbox.isChecked()),
            "add_outer_walls":{
                    "horizontal":  bool(self.add_horizontal_walls.isChecked()),
                    "vertical":  bool(self.add_vertical_walls.isChecked()),
                },
            "blend_enabled": bool(blend_enabled),
            "blend_harshness": blend_harshness,
            "transition_layer": transition_layer_margins,
            "blend_shrink_factors": {
                "x": x_shrink,
                "y": y_shrink
            },
            "tiles_per_phase_x": self.tiles_per_phase_x,
            "tiles_per_phase_y": self.tiles_per_phase_y,
            "phase_map": phase_map.tolist(),
            "blend_mask": blend_mask.tolist(),
            "blended_phases_map": blended_phases_serialized,
            "blend_directions": [[list(d) for d in row] for row in direction_map],
        }
        
    def collect_and_close(self):
        if self.run_button.isEnabled():  
            self.settings = self.collect_settings()
            self.close()
def run_input_gui():
    app = QApplication(sys.argv)
    window = MainApp()
    uniform_font = QFont("Arial", 11)
    uniform_font.setItalic(False)
    uniform_font.setBold(False)
    app.setFont(uniform_font)

    def on_confirm():
        settings = window.collect_settings()
        if not settings:
            return  # prevents crash if something failed
        with open("settings.json", "w") as f:
            json.dump(settings, f, indent=2)
        window.close()

    window.run_button.setText("Save and Close")
    window.run_button.clicked.disconnect()
    window.run_button.clicked.connect(on_confirm)

    window.show()
    app.exec_()
############################################################
#POSTPROCESSING data analysis
#the gradient frequencies interpolation    
def interpolate_frequencies(frequencies1, frequencies2, alpha):
    return {
        k: frequencies1.get(k, 0) * (1 - alpha) + frequencies2.get(k, 0) * alpha
        for k in set(frequencies1) | set(frequencies2)
    }    
    
#analyzing the actual achieved frequencies 
def analyze_tile_frequency_matrix_with_blending(matrix, row_frequencies, verbose=False):
    total_error = 0
    row_errors = []

    for y in range(matrix.shape[0]):
        actual_row = matrix[y, :]
        unique, counts = np.unique(actual_row, return_counts=True)
        actual_counts = dict(zip(unique, counts))
        total_tiles = matrix.shape[1]
        # Normalizing actual frequencies to percentages
        actual_perc = {k: (actual_counts.get(k, 0) / total_tiles) * 100 for k in [0, 2, 3, 4]}

        #Target frequencies for the row vs the output -> error
        target = row_frequencies[y]
        expected_perc = {k: target.get(k, 0) * 100 for k in [0, 2, 3, 4]}
        row_error = max( abs(actual_perc.get(k, 0) - expected_perc[k]) / max(expected_perc[k], 1e-6) for k in expected_perc)
        row_errors.append(row_error)
        total_error += row_error

    avg_error = total_error / matrix.shape[0]

    return avg_error, row_errors

def compute_avg_coordination_per_phase(frequency_matrix, phase_map, phases):
    phase_data = {}
    for y in range(frequency_matrix.shape[0]):
        for x in range(frequency_matrix.shape[1]):
            idx = phase_map[y][x]
            if idx not in phase_data:
                phase_data[idx] = []
            phase_data[idx].append(frequency_matrix[y, x])
    results = []
    for i, phase in enumerate(phases):
        avg = np.mean(phase_data.get(i, []))
        results.append({
            "phase": f"Phase {i + 1}",
            "target": phase["coordination_target"],
            "actual": round(avg, 2)
        })
    return results
#######################################################################################################################################################
#POSTPROCESSING images of the results visualisation, asinging points from centerlines, creating polygons
def create_tile_geometry(tile_type, tile_size=1.0):
    center = tile_size / 2
    wall_points = {
        "L": (0, center),
        "R": (tile_size, center),
        "T": (center, tile_size),
        "B": (center, 0)
    }
    diagonal_lines = {
        "tile_2_LT": LineString([(0, center), (center, tile_size)]),
        "tile_2_LB": LineString([(0, center), (center, 0)]),
        "tile_2_RT": LineString([(tile_size, center), (center, tile_size)]),
        "tile_2_RB": LineString([(tile_size, center), (center, 0)])
    }
    connection_sets = {
        "tile_0": [],
        "tile_4": ["L", "R", "T", "B"],
        "tile_3_noL": ["R", "T", "B"],
        "tile_3_noR": ["L", "T", "B"],
        "tile_3_noT": ["L", "R", "B"],
        "tile_3_noB": ["L", "R", "T"],
        "tile_2_LR": ["L", "R"],
        "tile_2_TB": ["T", "B"],
    }
    if tile_type in diagonal_lines:
        return [diagonal_lines[tile_type]]
    # Default case (non-diagonal)
    lines = []
    for conn in connection_sets.get(tile_type, []):
        lines.append(LineString([wall_points[conn], (center, center)]))
    return lines

def render_tile_svg_group(tile_lines, tile_size=1.0, tile_type="", flip_x=True, invert_y=True) -> svgwrite.container.Group:
    group = svgwrite.container.Group()
    group.add(
        svgwrite.shapes.Rect(insert=(0, 0), size=(tile_size, tile_size), fill="#ccc")
    )
    p = Path(stroke='black', stroke_width=0.05, fill='none')
    added = False  
    
    fx = (lambda x: tile_size - x) if flip_x else (lambda x: x)
    fy = (lambda y: tile_size - y) if invert_y else (lambda y: y)

    for line in tile_lines:
        coords = list(line.coords)
        if len(coords) >= 2:
            p.push(f'M {fx(coords[0][0])},{fy(coords[0][1])}')
            for pt in coords[1:]:
                p.push(f'L {fx(pt[0])},{fy(pt[1])}')
            added = True
    if added:
        group.add(p)
    return group

def precompute_tile_svgs(tile_types, tile_size=1.0):
    tile_svgs = {}
    for tile_type in tile_types:
        tile_lines = create_tile_geometry(tile_type, tile_size)
        group = render_tile_svg_group(tile_lines, tile_size, tile_type, flip_x=False, invert_y=False)
        tile_svgs[tile_type] = group
    return tile_svgs

def get_tile_key(tile, chosen_sum):
        connections = [tile[1][0], tile[2][1], tile[1][2], tile[0][1]]  # [L, B, R, T]
        if chosen_sum == 0:
            return "tile_0"
        if chosen_sum == 4:
            return "tile_4"
        if chosen_sum == 3:
            three_map = {
                (0, 1, 1, 1): "noL", (1, 0, 1, 1): "noB",
                (1, 1, 0, 1): "noR", (1, 1, 1, 0): "noT"
            }
            return f"tile_3_{three_map.get(tuple(connections), '')}"
        if chosen_sum == 2:
            pair_map = {
                (1, 0, 1, 0): "LR", (0, 1, 0, 1): "TB",
                (1, 1, 0, 0): "LB", (1, 0, 0, 1): "LT",
                (0, 1, 1, 0): "RB", (0, 0, 1, 1): "RT"
            }
            return f"tile_2_{pair_map.get(tuple(connections), '')}"
        return None
#L - left, R - right, T - top, B - bottom, The tile 0 is a special case with no connections, tile 4 is a full tile with all connections,
# tile 3 for the ease of distinction is described only by the connection that is missing, so tile 3 with no left connection is tile_3_noL,
# tile 2 is described by the connections that are present, so tile_2_LR is a tile with left and right connections

def get_offset(include_wall):
    return 1 if include_wall else 0

def save_final_matrix_as_svg(
    grid, tile_frequency_matrix, phase_map, blend_zones, settings,
    filename="final_matrix.svg", tile_size=1.0, annotate=True,
    tile_svgs=None
):
    assert tile_svgs is not None, "You must pass tile_svgs (from precompute_tile_svgs)"
    
    add_vertical = settings.get("add_outer_walls", {}).get("vertical", False)
    add_horizontal = settings.get("add_outer_walls", {}).get("horizontal", False)
    phase_rows = settings["phase_layout"]["rows"]
    phase_cols = settings["phase_layout"]["cols"]
    grid_width = settings["grid_width"]
    grid_height = settings["grid_height"]

    total_rows = len(grid)
    total_cols = len(grid[0])

    # Wall padding offsets (number of tiles before phase grid starts)
    x_offset = 1 if add_vertical else 0
    y_offset = 1 if add_horizontal else 0

    dwg = svgwrite.Drawing(
        filename,
        size=(f"{total_cols * tile_size}cm", f"{total_rows * tile_size}cm"),
        viewBox=f"0 0 {total_cols * tile_size} {total_rows * tile_size}"
    )
    center_x = total_cols * tile_size / 2
    center_y = total_rows * tile_size / 2

    main_group = dwg.g(transform=f"rotate(180,{center_x},{center_y})")
    dwg.add(dwg.rect(
        insert=(0, 0),
        size=(total_cols * tile_size, total_rows * tile_size),
        fill='white'
    ))

    for y in range(total_rows):
        for x in range(total_cols):
            tx = x * tile_size
            ty = (total_rows - 1 - y) * tile_size

            if (y, x) in blend_zones:
                dwg.add(dwg.rect(
                    insert=(tx, ty),
                    size=(tile_size, tile_size),
                    fill='orange',
                    fill_opacity=0.3
                ))

            tile = grid[y][x]
            freq = tile_frequency_matrix[y][x]
            key = get_tile_key(tile, freq)

            if key in tile_svgs:
                tile_group = tile_svgs[key].copy()
                tile_group.translate(tx, ty)
                dwg.add(tile_group)

    # Annotation (phase lines, blend edges, border box)
    if annotate:
        rows_per_phase = grid_height // phase_rows
        cols_per_phase = grid_width // phase_cols

        cyan_style = {'stroke': 'cyan', 'stroke_width': tile_size * 0.1}
        orange_style = {'stroke': 'orange', 'stroke_width': tile_size * 0.15, 'fill': 'none'}

        # Phase division lines
        for i in range(1, phase_rows):
            y = (i * rows_per_phase + y_offset) * tile_size
            dwg.add(dwg.line(
                start=(x_offset * tile_size, y),
                end=((x_offset + grid_width) * tile_size, y),
                **cyan_style
            ))

        for i in range(1, phase_cols):
            x = (i * cols_per_phase + x_offset) * tile_size
            dwg.add(dwg.line(
                start=(x, y_offset * tile_size),
                end=(x, (y_offset + grid_height) * tile_size),
                **cyan_style
            ))

        # Outline each blend tile edge (if touching non-blend)
        for y in range(total_rows):
            for x in range(total_cols):
                if (y, x) not in blend_zones:
                    continue
                tx = (x + x_offset) * tile_size
                ty = (y + y_offset) * tile_size

                if y == 0 or (y-1, x) not in blend_zones:
                    dwg.add(dwg.line(start=(tx, ty), end=(tx + tile_size, ty), **orange_style))  # Top
                if y == total_rows - 1 or (y+1, x) not in blend_zones:
                    dwg.add(dwg.line(start=(tx, ty + tile_size), end=(tx + tile_size, ty + tile_size), **orange_style))  # Bottom
                if (y, x-1) not in blend_zones:
                    dwg.add(dwg.line(start=(tx, ty), end=(tx, ty + tile_size), **orange_style))  # Left
                if (y, x+1) not in blend_zones:
                    dwg.add(dwg.line(start=(tx + tile_size, ty), end=(tx + tile_size, ty + tile_size), **orange_style))  # Right

        # Outer border box
        dwg.add(dwg.rect(
            insert=(x_offset * tile_size, y_offset * tile_size),
            size=(grid_width * tile_size, grid_height * tile_size),
            fill='none', stroke='purple', stroke_width=tile_size * 0.1
        ))
    dwg.add(main_group)
    dwg.save()
    
def clip_to_tile_bounds(geometry, point, tile_size_mm):
    tile_x = int(point[0] // tile_size_mm)
    tile_y = int(point[1] // tile_size_mm)
    tile_bounds = box(
        tile_x * tile_size_mm,
        tile_y * tile_size_mm,
        (tile_x + 1) * tile_size_mm,
        (tile_y + 1) * tile_size_mm
    )
    return geometry.intersection(tile_bounds)

def diagonal_convex_strut(wall_a, wall_b, center, width):
    def offset_point(from_pt, to_pt, offset):
        dx = to_pt[0] - from_pt[0]
        dy = to_pt[1] - from_pt[1]
        length = (dx**2 + dy**2) ** 0.5
        if length == 0:
            return (from_pt, from_pt)
        ux, uy = dx / length, dy / length
        nx, ny = -uy * offset, ux * offset
        # Return two points offset to left and right of line along the bounary walls
        return ((from_pt[0] + nx, from_pt[1] + ny),
                (from_pt[0] - nx, from_pt[1] - ny))
    # Offset the struts from each wall point toward the center
    a1, a2 = offset_point(wall_a, center, width / 2)
    b2, b1 = offset_point(wall_b, center, width / 2)
    # Return (a1 → b1 → b2 → a2) to correctly connect the points into non crossed shape
    return Polygon([a1, b1, b2, a2])

def debug_save_buffered_svg(buffered_shapes, out_path, tile_size_mm):
    dwg = svgwrite.Drawing(out_path, profile='tiny')

    all_bounds = [shape.bounds for shape in buffered_shapes if not shape.is_empty]
    if not all_bounds:
        print("[DEBUG] No shapes to draw.")
        return

    min_x = min(b[0] for b in all_bounds)
    min_y = min(b[1] for b in all_bounds)
    max_x = max(b[2] for b in all_bounds)
    max_y = max(b[3] for b in all_bounds)
    width = max_x - min_x
    height = max_y - min_y

    dwg.viewbox(min_x, min_y, width, height)

    #Apply 180° rotation around center of bounding box ===
    rotation_center_x = min_x + width / 2
    rotation_center_y = min_y + height / 2
    transform_string = f"rotate(180,{rotation_center_x},{rotation_center_y})"

    main_group = dwg.g(transform=transform_string)

    # Background
    main_group.add(dwg.rect(
        insert=(min_x, min_y),
        size=(width, height),
        fill='white'
    ))

    for shape in buffered_shapes:
        if shape.is_empty:
            continue
        if isinstance(shape, (Polygon, MultiPolygon)):
            polys = [shape] if isinstance(shape, Polygon) else shape.geoms
            for poly in polys:
                minx, miny, maxx, maxy = poly.bounds
                is_tile_box = (
                    abs(maxx - minx - tile_size_mm) < 1e-6 and
                    abs(maxy - miny - tile_size_mm) < 1e-6
                )
                fill_color = 'black'
                stroke_color = 'black' if is_tile_box else 'black'

                main_group.add(dwg.polygon(
                    points=list(poly.exterior.coords),
                    fill=fill_color,
                    fill_opacity=0.4,
                    stroke=stroke_color,
                    stroke_width=0.02
                ))

    dwg.add(main_group)
    dwg.save()
    #print(f"[DEBUG] Buffered shapes saved to: {out_path}")
    
    
def line_to_polygon(start, end, thickness):
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = (dx**2 + dy**2) ** 0.5
    if length == 0:
        return Polygon()
    ux, uy = dx / length, dy / length
    nx, ny = -uy * thickness / 2, ux * thickness / 2
    return Polygon([
        (start[0] + nx, start[1] + ny),
        (start[0] - nx, start[1] - ny),
        (end[0] - nx, end[1] - ny),
        (end[0] + nx, end[1] + ny)
    ])


def clean_and_union_shapes(shapes, simplify_tolerance=0.001):
    valid_shapes = [s.buffer(0) for s in shapes if s.is_valid and not s.is_empty]
    filtered = [s for s in valid_shapes if s.area > 1e-6]

    combined = unary_union(filtered)

    if isinstance(combined, (Polygon, MultiPolygon)):
        combined = combined.simplify(simplify_tolerance, preserve_topology=True)
        combined = combined.buffer(0)

    if not combined.is_valid:
        print("[WARNING] Final geometry is still invalid, attempting to fix...")
        combined = make_valid(combined)
    return combined

def svg_to_stl_from_grid(
    grid,
    tile_frequency_matrix,
    stl_path="final_matrix.stl",
    tile_size_mm=1.0,
    line_thickness=0.2,
    extrusion_height=1.0,
    known_tile_types=None
):
    print("[STL] Generating STL from svg...")
    buffered_shapes = []
    grid_height = len(grid)
    grid_width = len(grid[0])

    for y in range(grid_height):
        for x in range(grid_width):
            tile = grid[y][x]
            freq = tile_frequency_matrix[y][x]
            tile_key = get_tile_key(tile, freq)

            if tile_key not in known_tile_types:
                continue  

            tile_lines = create_tile_geometry(tile_key, tile_size_mm)
            tile_shapes = []
            for line in tile_lines:
                start, end = list(line.coords)
                diagonal_wall_pairs = {
                    "LT": ("L", "T"),
                    "LB": ("L", "B"),
                    "RT": ("R", "T"),
                    "RB": ("R", "B"),
                }
                wall_points = {
                    "L": (0, tile_size_mm / 2),
                    "R": (tile_size_mm, tile_size_mm / 2),
                    "T": (tile_size_mm / 2, tile_size_mm),
                    "B": (tile_size_mm / 2, 0)
                }
                if tile_key.startswith("tile_2_") and tile_key[-2:] in diagonal_wall_pairs:
                    wall_a_name, wall_b_name = diagonal_wall_pairs[tile_key[-2:]]
                    wall_a = wall_points[wall_a_name]
                    wall_b = wall_points[wall_b_name]
                    center = (tile_size_mm / 2, tile_size_mm / 2)
                    shape = diagonal_convex_strut(
                        wall_a, wall_b, center, line_thickness
                    )
                    shape = clip_to_tile_bounds(shape, ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2), tile_size_mm)
                else:
                    shape = line_to_polygon(start, end, line_thickness)
                    shape = clip_to_tile_bounds(shape, ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2), tile_size_mm)
                tile_shapes.append(shape)
            tile_combined = unary_union(tile_shapes)

            #translate the entire tile
            translated = affinity.translate(
                tile_combined,
                xoff = round(x * tile_size_mm, 4),
                yoff = round((grid_height - 1 - y) * tile_size_mm, 4)
            )
            buffered_shapes.append(translated)      
    if not buffered_shapes:
        print("[STL ERROR] No shapes buffered.")
        return
    combined = clean_and_union_shapes(buffered_shapes)
    extruded_meshes = []
    if isinstance(combined, Polygon):
        if not combined.is_valid:
            print("[FIX] Invalid polygon detected — attempting to repair...")
            combined = make_valid(combined)
        extruded = trimesh.creation.extrude_polygon(combined, extrusion_height, engine="earcut")
        if extruded.volume > 0:
            extruded_meshes.append(extruded)
    elif isinstance(combined, MultiPolygon):
        for poly in combined.geoms:
            if not poly.is_valid:
                print("[FIX] Invalid sub-polygon — repairing...")
                poly = make_valid(poly)
            extruded = trimesh.creation.extrude_polygon(poly, extrusion_height, engine="earcut")
            if extruded.volume > 0:
                extruded_meshes.append(extruded)
    if extruded_meshes:
        final_mesh = trimesh.util.concatenate(extruded_meshes)
        final_mesh.export(stl_path, file_type='stl')
        print(f"[STL] Saved to: {stl_path}")
    else:
        print("[STL ERROR] No meshes created.")
    if final_mesh.is_watertight:
        print("[STL] Mesh is watertight.")
    else:
        print("[STL] Mesh is watertight.")
        #print("[STL WARNING] Mesh is NOT watertight. Likely to fail print.")
    debug_svg_path = stl_path.replace(".stl", "_debug.svg")
    debug_save_buffered_svg(buffered_shapes, debug_svg_path, tile_size_mm)    
    return buffered_shapes

        
def generate_inverse_stl(
    grid,
    buffered_shapes,
    out_path,
    tile_size_mm,
    extrusion_height,
    add_outer_walls=None
):
    print("[INVERSE STL] Generating inverse...")
    grid_height = len(grid)
    grid_width = len(grid[0])
    if add_outer_walls is None:
        add_outer_walls = {"horizontal": False, "vertical": False}
    #Print size
    base_width_mm = grid_width * tile_size_mm
    base_height_mm = grid_height * tile_size_mm
    min_x = 0
    min_y = 0
    max_x = base_width_mm
    max_y = base_height_mm
    if add_outer_walls.get("vertical", False):
        min_x += tile_size_mm / 2  # trim left
        max_x -= tile_size_mm / 2  # trim right
    if add_outer_walls.get("horizontal", False):
        min_y += tile_size_mm / 2  # trim bottom
        max_y -= tile_size_mm / 2  # trim top
    bounding_box = box(min_x, min_y, max_x, max_y)
    #print(f"[INVERSE] Bounding box: {bounding_box.bounds}")
    solid = unary_union(buffered_shapes)
    inverse_shape = bounding_box.difference(solid)
    extruded_meshes = []
    if isinstance(inverse_shape, Polygon):
        extruded_meshes.append(trimesh.creation.extrude_polygon(inverse_shape, extrusion_height, engine="earcut"))
    elif isinstance(inverse_shape, MultiPolygon):
        for poly in inverse_shape.geoms:
            extruded_meshes.append(trimesh.creation.extrude_polygon(poly, extrusion_height, engine="earcut"))
    if extruded_meshes:
        final_mesh = trimesh.util.concatenate(extruded_meshes)
        final_mesh.export(out_path)
        print(f"[INVERSE STL] Saved to: {out_path}")
    else:
        print("[INVERSE STL ERROR] No inverse mesh created.")
#######################################################################################################################################################
#POSTPROCESSING graphs visualisation
def plot_iteration_error_summary(summary_log, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    errors = [entry["error"] for entry in summary_log]
    iterations = list(range(1, len(errors) + 1))
    plt.figure(figsize=(12, 6))
    plt.plot(iterations, errors, color='black', linestyle='-', marker='o', label="Error")

    top3 = sorted(enumerate(errors), key=lambda x: x[1])[:3]
    colors = ["gold", "silver", "peru"]
    labels = ["1", "2", "3"]

    for rank, (idx, err) in enumerate(top3):
        plt.plot(iterations[idx], err, marker='o', markersize=10, color=colors[rank])
        plt.text(iterations[idx], err + 0.005, labels[rank], ha='center', va='bottom',
                 fontsize=12, fontweight='bold', color=colors[rank])

    plt.xlabel("Itteration number")
    plt.ylabel("Average Coordination Error")
    plt.title("Coordination Error across 50 Runs\n(Top 3 Highlighted)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, "iteration_error_summary.png"))
    plt.close()
        
def plot_tile_frequencies_grid(tile_frequency_matrix, phase_map, target_tile_frequencies,
                                phase_rows, phase_cols, output_dir, blend_mask=None):
    os.makedirs(output_dir, exist_ok=True)

    total_phases = phase_rows * phase_cols
    fig, axes = plt.subplots(phase_rows, phase_cols, figsize=(phase_cols * 4, phase_rows * 4))
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    
    phase_tile_dict = {i: [] for i in range(total_phases)}

    for y in range(tile_frequency_matrix.shape[0]):
        for x in range(tile_frequency_matrix.shape[1]):
            if blend_mask is not None and blend_mask[y][x]:
                continue  
            phase_idx = phase_map[y][x]
            val = tile_frequency_matrix[y][x]
            if phase_idx in phase_tile_dict:
                phase_tile_dict[phase_idx].append(val)
    for i in range(total_phases):
        ax = axes[i]
        phase_tiles = phase_tile_dict[i]
        total = len(phase_tiles)
        if total == 0:
            ax.set_visible(False)
            continue
        unique, counts = np.unique(phase_tiles, return_counts=True)
        actual_freqs = dict(zip(unique, counts))
        actual_perc = {r: (actual_freqs.get(r, 0) / total) * 100 for r in [0, 2, 3, 4]}
        actual_coord = sum((actual_freqs.get(r, 0) / total) * r for r in [0, 2, 3, 4])

        target_freqs = target_tile_frequencies[i]
        target_avg = {r: target_freqs.get(r, 0.0) * 100 for r in [0, 2, 3, 4]}
        target_coord = sum(target_freqs.get(r, 0.0) * r for r in [0, 2, 3, 4])

        labels = ["R=0", "R=2", "R=3", "R=4"]
        x = np.arange(len(labels))
        width = 0.35

        ax.bar(x - width / 2, [target_avg[r] for r in [0, 2, 3, 4]], width, label="Target", color="skyblue")
        ax.bar(x + width / 2, [actual_perc[r] for r in [0, 2, 3, 4]], width, label="Actual", color="coral")

        ax.set_title(f"Phase {i + 1}\nTarget Coord: {target_coord:.2f} | Actual: {actual_coord:.2f}", fontsize=10)
        ax.set_ylim(0, 100)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)
        if i == 0:
            ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tile_frequency_accuracy_per_each_phase_on_grid_layout.png"))
    plt.close()
#########################################################################################################################################################
#VGA - check the current location, look up expected outcome, search the history of precious tiles, check the conitnuity contraints, assign the new tile, store the data and pass it on until the grid is filled
def run_tile_generation_VGA(adjustment_mode, weight, phases,
                            phase_map, blend_mask, blend_directions,
                            blended_phases_map, tiles_x, tiles_y):
    grid = np.zeros((GRID_HEIGHT, GRID_WIDTH, TILE_SIZE, TILE_SIZE), dtype=int)
    saved_matrix = np.zeros((GRID_HEIGHT * TILE_SIZE, GRID_WIDTH * TILE_SIZE), dtype=int)
    frequency_matrix = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=int)
    tile_dict_matrix = [[None for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]

    # Extract enforced phase rules
    phase_enforced_rules = {
        i: p.get("connectivity_rules", {}) for i, p in enumerate(phases)
    }

    # Precompute blend weights per tile
    blend_weight_map = [[{} for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            if blend_mask[y, x]:
                blend_weight_map[y][x] = get_rule_blend_weights(
                    x, y,
                    neighbor_phases=blended_phases_map[y][x],
                    phase_map=phase_map,
                    tiles_per_phase_x=tiles_x,
                    tiles_per_phase_y=tiles_y,
                    direction_map=blend_directions
                )

    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            constraints = generation_demo_1_grid_constraints(x, y, grid)
            current_phase = phase_map[y][x]

            #Rile probabilities & allowed types
            if blend_mask[y, x] and isinstance(blend_directions[y][x], (list, set)):
                adjusted_probs, possibility_matrix = get_effective_transition_tile_policy(
                    x, y,
                    phase_map=phase_map,
                    blended_phases_map=blended_phases_map,
                    phases=phases,
                    constraints=constraints,
                    direction_map=blend_directions,
                    frequency_matrix=frequency_matrix,
                    blend_mask=blend_mask
                )
            else:
                adjusted_probs, possibility_matrix = get_effective_tile_policy(
                    x, y,
                    phases, phase_map,
                    frequency_matrix,
                    adjustment_mode, weight,
                    blend_mask
                )

            # Tile type with rule filtering 
            chosen_sum = generation_demo_2_filter_tile_options(
                constraints=constraints,
                grid=grid,
                x=x, y=y,
                phase_map=phase_map,
                blended_phases_map=blended_phases_map,
                blend_mask=blend_mask,
                tile_matrix=possibility_matrix,
                frequencies=adjusted_probs,
                phase_enforced_rules=phase_enforced_rules,
                blend_weight_map=blend_weight_map,
                direction_map=blend_directions,
                tiles_per_phase_x=tiles_x,
                tiles_per_phase_y=tiles_y
            )

            # Assign connections 
            active_connections = generation_demo_3_assign_remaining_connections(
                grid=grid,
                x=x, y=y,
                chosen_sum=chosen_sum,
                constraints=constraints,
                phase_rules=phase_enforced_rules,
                phase=current_phase,
                blend_mask=blend_mask,
                blended_phases_map=blended_phases_map,
                blend_weight_map=blend_weight_map
            )

            # Build tile from connection pattern with the steps
            tile, confirmed_sum = generation_demo_4_construct_tile(active_connections, chosen_sum)

            # Write results
            grid[y, x] = tile
            saved_matrix[y * TILE_SIZE:(y + 1) * TILE_SIZE, x * TILE_SIZE:(x + 1) * TILE_SIZE] = tile
            frequency_matrix[y, x] = confirmed_sum
            tile_dict_matrix[y][x] = {
                "tile": tile,
                "active_connections": active_connections,
                "sum": confirmed_sum
            }

    return grid, saved_matrix, frequency_matrix, tile_dict_matrix
#######################################################################################################################################################
#the optimization call where we itterate over multiple attempts to obtain the best results in a reasonable amount of time 
def optimize_itterate_strategy(phases, phase_map, blend_mask, blend_directions, blended_phases_map,
                             grid_width, grid_height, tile_size,tiles_x, tiles_y, iterations=10):
    summary_log = []
    modes = ['greedy', 'balanced', 'conservative', 'sharp', 'lenient']
    mode_weights = {
        'greedy': 1.6,
        'balanced': 1.0,
        'conservative': 0.4,
        'sharp': 2.2,
        'lenient': 0.2
    }
    global GRID_WIDTH, GRID_HEIGHT, TILE_SIZE
    GRID_WIDTH = grid_width
    GRID_HEIGHT = grid_height
    TILE_SIZE = tile_size
    results = {}
    step = 0.3
    run_id = 0
    for iteration in range(iterations):
        if iteration >=1:
            print(f"[Started Itteration {iteration:02d}] 5 correction weights were set based on the memory of previous itterations")
        else:
            print(f"[Started Itteration {iteration:02d}] 5 correction weights were set based on default settings")
        for mode in modes:
            run_id += 1
            weight = mode_weights[mode] 
            print(f"[Started Run {run_id:02d}] Mode: {mode}, Weight: {weight:.2f}")
            grid, matrix, frequency_matrix, tile_dict_matrix = run_tile_generation_VGA(
                adjustment_mode=mode,
                weight=weight,
                phases=phases,
                phase_map=phase_map,
                blend_mask=blend_mask,
                blended_phases_map=blended_phases_map,
                blend_directions=blend_directions,
                tiles_x=tiles_x, 
                tiles_y=tiles_y
            )
            # Aggregate per-phase tile frequencies
            error = 0
            phase_errors = []
            num_phases = len(phases)
            for phase_idx in range(num_phases):
                coords = [(y, x) for y in range(GRID_HEIGHT) for x in range(GRID_WIDTH)
                            if phase_map[y][x] == phase_idx and not blend_mask[y][x]]
                if not coords:
                    phase_errors.append(1.0)
                    error += 1.0
                    continue
                counts = {}
                for y, x in coords:
                    t = frequency_matrix[y][x]
                    counts[t] = counts.get(t, 0) + 1
                total = sum(counts.values())
                if total == 0:
                    phase_errors.append(1.0)
                    error += 1.0
                    continue
                actual_freqs = {k: v / total for k, v in counts.items()}
                target = {int(k): v for k, v in phases[phase_idx]["tile_frequencies"].items()}
                err = sum(abs(actual_freqs.get(k, 0) - target.get(k, 0)) for k in [0, 2, 3, 4])
                phase_errors.append(err)
                error += err
            avg_error = error / num_phases
            results[run_id] = {
                "error": avg_error,
                "grid": grid,
                "matrix": matrix,
                "tile_dict_matrix": tile_dict_matrix,
                "frequency_matrix": frequency_matrix,
                "mode": mode,
                "weight": weight
            }
            summary_log.append({
                "run_id": run_id,
                "iteration": iteration + 1,
                "mode": mode,
                "weight": weight,
                "error": avg_error
            })
        #  focus opn best performing adjustment gradient
        best_result = sorted(results.values(), key=lambda v: v["error"])[0]
        best_weight = best_result["weight"]
        base = best_weight
        step = max(0.1, step * 0.8)  # orient the new adjsutments focus aroumd the best perfoming weight and decrease the deviation to search for most optimal setting
        mode_offsets = {
            'sharp':     2.0 * step,
            'greedy':    1.0 * step,
            'balanced':  0.0,
            'conservative': -1.0 * step,
            'lenient':   -2.0 * step
        }
        for mode in mode_weights:
            mode_weights[mode] = max(0.01, base + mode_offsets[mode])
    # Sort by error
    top_runs = sorted(results.items(), key=lambda kv: kv[1]["error"])[:3]
    print("\nBest 3 Runs:")
    for i, (run_id, result) in enumerate(top_runs, 1):
        print(f"{i}. Run {run_id:02d} | Correction weight: {result['weight']:.2f} | Overall average Error: {result['error']:.4f}")
    return top_runs, summary_log, grid

def save_blend_mask_image(blend_mask, path="blend_mask_debug.png"):
    plt.figure(figsize=(10, 10))
    plt.imshow(blend_mask, cmap='Oranges', interpolation='none')
    plt.title("Blend Mask (True = Transition Zone)")
    plt.savefig(path)
    plt.close()
    print(f"[DEBUG] Blend mask visualization saved to {path}")
    
def save_phase_blend_overlay(phase_map, blend_mask, path="phase_blend_overlay.png"):
    overlay = np.array(phase_map, dtype=float)
    overlay[blend_mask] = -1  # Use -1 for blend zone
    plt.figure(figsize=(10, 10))
    plt.imshow(overlay, cmap='tab20', interpolation='none')
    plt.title("Phases + Blend Mask (-1 = blend zone)")
    plt.colorbar()
    plt.savefig(path)
    plt.close()
    print(f"[DEBUG] Phase/blend overlay saved to {path}")
######################################################################################################################################################
#The main call how assign the stored data, what I visualize and plot, how long it takes to process the user inputs, debuging/ informative calls of which process is happening
def main_simulation_logic(settings):
    print("Got user input's, running simulation...")
    start_time = time.time()    
    GRID_WIDTH = settings["grid_width"]
    GRID_HEIGHT = settings["grid_height"]
    OUTPUT_PATH = settings["output_path"]
    num_samples = settings["num_samples"]
    phases = settings["phases"]
    transition = settings.get("transition_layer", {})
    x_shrink = settings.get("blend_shrink_factors", {}).get("x", 0)
    y_shrink = settings.get("blend_shrink_factors", {}).get("y", 0)
    phase_rows = settings["phase_layout"]["rows"]
    phase_cols = settings["phase_layout"]["cols"]
    clear_output_folder(OUTPUT_PATH, exceptions=["input_settings.json"])
    phase_map = np.array(settings["phase_map"])
    blend_mask = np.array(settings["blend_mask"])
    blend_directions = [
        [set(dirs) for dirs in row] for row in settings["blend_directions"]
    ]
    blended_phases_map = [
    [set(cell) for cell in row] for row in settings.get("blended_phases_map", [[]])
    ]
    
    # np.savetxt(os.path.join(OUTPUT_PATH, "blend_mask.csv"), blend_mask.astype(int), fmt="%d", delimiter=",")
    # with open(os.path.join(OUTPUT_PATH, "blended_phases_map.csv"), "w", newline="") as f:
    #     writer = csv.writer(f)
    #     for row in blended_phases_map:
    #         writer.writerow([";".join(map(str, sorted(cell))) if cell else "" for cell in row])
    # np.savetxt(os.path.join(OUTPUT_PATH, "phase_map.csv"),phase_map.astype(int), fmt="%d", delimiter=",")
    # # --- Save direction_map as CSV (string representation of sets) ---
    # with open(os.path.join(OUTPUT_PATH, "blend_directions.csv"), "w", newline='') as f:
    #     writer = csv.writer(f)
    #     for row in blend_directions:
    #         writer.writerow([",".join(sorted(d)) if d else "" for d in row])    
    print("Running the simmulaiton creating the geometries")
    for phase in settings["phases"]:
        if "tile_frequencies" in phase:
            phase["tile_frequencies"] = {int(k): v for k, v in phase["tile_frequencies"].items()}
    
    Tile_generation_start = time.time()
    top_runs, summary_log,grid  = optimize_itterate_strategy(
    phases=phases,
    phase_map=phase_map,
    blend_mask=blend_mask,
    blend_directions=blend_directions,
    blended_phases_map=blended_phases_map,
    grid_width=GRID_WIDTH,
    grid_height=GRID_HEIGHT,
    tile_size=TILE_SIZE,
    tiles_x=settings["tiles_per_phase_x"],
    tiles_y=settings["tiles_per_phase_x"],
    iterations=settings["num_iterations"]
    )
    # grid_path = os.path.join(OUTPUT_PATH, f"grid_path.csv")
    # pd.DataFrame(grid).to_csv(grid_path, index=False)
    Tile_generation_end = time.time()
    print(f"Stage 1 took {Tile_generation_end - Tile_generation_start:.4f} seconds")
    with open(os.path.join(OUTPUT_PATH, "input_settings.json"), "w") as f:
        json.dump(settings, f, indent=2)
    time.sleep(2)
    
    tile_types = [
    "tile_0", "tile_2_LR", "tile_2_TB", "tile_2_LB", "tile_2_LT",
    "tile_2_RB", "tile_2_RT", "tile_3_noL", "tile_3_noB",
    "tile_3_noR", "tile_3_noT", "tile_4"
    ]

    tile_svgs = precompute_tile_svgs(tile_types,settings["tile_size_mm"])
    blend_zones = {
    (y, x) for y in range(blend_mask.shape[0]) for x in range(blend_mask.shape[1]) if blend_mask[y, x]
}
    image_generation_start = time.time()
    for i, (run_id, result) in enumerate(top_runs[:num_samples], start=1):
        tile_dict_matrix_with_boundaries = generate_boundaries(
            saved_matrix=result["tile_dict_matrix"],
            grid_width=GRID_WIDTH,
            grid_height=GRID_HEIGHT,
            add_horizontal=settings.get("add_outer_walls", {}).get("horizontal", True),
            add_vertical=settings.get("add_outer_walls", {}).get("vertical", True)
        )
        matrix = np.array([
            [cell["sum"] if isinstance(cell, dict) else 0 for cell in row]
            for row in tile_dict_matrix_with_boundaries
        ])
        svg_path = os.path.join(OUTPUT_PATH, f"best_final_matrix_{i}.svg")
        svg_path_anotated = os.path.join(OUTPUT_PATH, f"best_final_matrix_anotated_{i}.svg")
        height_with_boundaries = len(tile_dict_matrix_with_boundaries)
        width_with_boundaries = len(tile_dict_matrix_with_boundaries[0])

        grid_with_boundaries = np.zeros(
            (height_with_boundaries, width_with_boundaries, TILE_SIZE, TILE_SIZE), dtype=int
        )

        for y in range(height_with_boundaries):
            for x in range(width_with_boundaries):
                cell = tile_dict_matrix_with_boundaries[y][x]
                if isinstance(cell, dict):
                    grid_with_boundaries[y, x] = cell["tile"]
        save_final_matrix_as_svg(
            grid=grid_with_boundaries,
            tile_frequency_matrix=matrix,
            phase_map=phase_map,
            blend_zones=blend_zones,
            settings=settings,
            filename=svg_path_anotated,
            tile_size=settings["tile_size_mm"],
            annotate=True,
            tile_svgs=tile_svgs
        )
        #matrix_csv_path = os.path.join(OUTPUT_PATH, f"best_final_matrix_{i}_frequencies.csv")
        #pd.DataFrame(matrix).to_csv(matrix_csv_path, index=False)
        #grid_csv_path = os.path.join(OUTPUT_PATH, f"best_grid_matrix_{i}.csv")
        expanded_rows = []
        for y in range(grid_with_boundaries.shape[0]):
            for x in range(grid_with_boundaries.shape[1]):
                tile = grid_with_boundaries[y, x]
                expanded_rows.append(tile.flatten().tolist())
        #grid_df = pd.DataFrame(expanded_rows)
        #grid_df.to_csv(grid_csv_path, index=False)
        #dict_matrix_csv_path = os.path.join(OUTPUT_PATH, f"best_final_matrix_{i}_dict_matrix.csv")
        #df_dict_matrix = pd.DataFrame([
        #     [cell["sum"] if isinstance(cell, dict) else "" for cell in row]
        #     for row in tile_dict_matrix_with_boundaries
        # ])
        #df_dict_matrix.to_csv(dict_matrix_csv_path, index=False)

        if(settings["stl"]):
            stl_path = os.path.join(OUTPUT_PATH, f"best_final_matrix_{i}.stl")
            buffered_shapes =svg_to_stl_from_grid(
                grid = grid_with_boundaries,
                tile_frequency_matrix=matrix,
                stl_path=stl_path,
                tile_size_mm=settings["tile_size_mm"],
                line_thickness=settings["line_thickness"],
                extrusion_height=settings["extrusion_height"],
                known_tile_types=tile_types
            )
            generate_inverse_stl(
                grid = grid_with_boundaries,
                buffered_shapes=buffered_shapes,
                out_path=os.path.join(os.path.dirname(stl_path),"inverse_" + os.path.basename(stl_path)),
                tile_size_mm=settings["tile_size_mm"],
                extrusion_height=settings["extrusion_height"],
                add_outer_walls={
                    "horizontal": settings.get("add_outer_walls", {}).get("horizontal", False),
                    "vertical": settings.get("add_outer_walls", {}).get("vertical", False)
                }
            )
    image_generation_end = time.time()
    if (settings["stl"]):
        print(f"SVG's and STl's generation took {image_generation_end -image_generation_start:.4f} seconds")
    else:
        print(f"SVG's generation took {image_generation_end -image_generation_start:.4f} seconds")
    time.sleep(2)
    
    Data_analysis_start = time.time()
    print("\nFinal Best Configuration Data Analysis:")
    
    target_frequencies_from_settings = [
    {int(k): v for k, v in phase.get("tile_frequencies", {}).items()}
    for phase in settings["phases"]]
    
    plot_iteration_error_summary(summary_log, OUTPUT_PATH)
    for i, (run_id, result) in enumerate(top_runs[:num_samples], start=1):
        output_dir_i = os.path.join(OUTPUT_PATH, f"sample_{i}")
        os.makedirs(output_dir_i, exist_ok=True)
        
        plot_tile_frequencies_grid(
            tile_frequency_matrix=result["frequency_matrix"],
            phase_map=phase_map,
            target_tile_frequencies=target_frequencies_from_settings,
            phase_rows=phase_rows,
            phase_cols=phase_cols,
            output_dir=output_dir_i,
            blend_mask=blend_mask
        )

    Data_analysis_end = time.time()
    print(f"Data analysis took {Data_analysis_end - Data_analysis_start:.4f} seconds")
    print("Simulation done. Results saved to:", OUTPUT_PATH)
    print("Total time taken:", time.time() - start_time)
#Once we run the code the interface will be called the settings will be saved, then passed over to the algorithm to create the desired sturctures over mutliple itterations and the results will be evaluated, the process will be explaied outptu messages
if __name__ == "__main__":
    run_input_gui()
    with open("settings.json", "r") as f:
        settings = json.load(f)
    # OUTPUT_PATH = settings["output_path"]
    # LOG_FILE = os.path.join(OUTPUT_PATH, "tile_policy_debug_log.csv")
    # if not os.path.exists(LOG_FILE):
    #     with open(LOG_FILE, mode='w', newline='') as f:
    #         writer = csv.writer(f)
    #         writer.writerow([
    #             "x", "y", "direction", "alpha",
    #             "phase_indices", "current_phase",
    #             "weights", "blended_freq", 
    #             "actual_counts", "adjusted_probs"
    #         ])
    main_simulation_logic(settings)
