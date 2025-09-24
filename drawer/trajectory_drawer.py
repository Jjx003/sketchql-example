#!/usr/bin/env python3
"""
Interactive trajectory drawing tool using pygame.

This tool allows users to draw trajectories for up to 4 different objects using
different colors. Trajectories are saved as JSON files that can be loaded by
the SketchQL model for similarity comparison.

Usage:
    python trajectory_drawer.py

Controls:
    - Left mouse button: Draw trajectory for current object
    - Right mouse button: Erase last point
    - 1, 2, 3, 4: Switch between objects (colors)
    - S: Save current trajectories
    - C: Clear all trajectories
    - R: Reset to start
    - ESC: Exit

Author: SketchQL Team
"""

import pygame
import json
import sys
from typing import Tuple
from datetime import datetime


class TrajectoryDrawer:
    """Interactive trajectory drawing tool with pygame."""
    
    def __init__(self, width: int = 800, height: int = 600):
        """Initialize the drawing tool.
        
        Args:
            width: Window width in pixels
            height: Window height in pixels
        """
        pygame.init()
        
        # Window setup
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("SketchQL Trajectory Drawer")
        
        # Colors for different objects
        self.colors = {
            0: (255, 0, 0),    # Red
            1: (0, 255, 0),    # Green  
            2: (0, 0, 255),    # Blue
            3: (255, 255, 0)   # Yellow
        }
        
        # Object names
        self.object_names = ["Object 1", "Object 2", "Object 3", "Object 4"]
        
        # Drawing state
        self.current_object = 0
        self.trajectories = {i: [] for i in range(4)}  # {object_id: [(x, y), ...]}
        self.drawing = False
        
        # UI colors
        self.bg_color = (240, 240, 240)
        self.text_color = (0, 0, 0)
        self.grid_color = (200, 200, 200)
        
        # Font setup
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
        # Grid settings
        self.grid_size = 20
        
    def draw_grid(self):
        """Draw a grid on the canvas."""
        ui_height = 80  # Match the UI height
        for x in range(0, self.width, self.grid_size):
            pygame.draw.line(self.screen, self.grid_color, (x, ui_height), (x, self.height), 1)
        for y in range(ui_height, self.height, self.grid_size):
            pygame.draw.line(self.screen, self.grid_color, (0, y), (self.width, y), 1)
    
    def draw_trajectories(self):
        """Draw all trajectories on the canvas."""
        ui_height = 80  # Match the UI height
        for obj_id, trajectory in self.trajectories.items():
            if len(trajectory) < 2:
                continue
                
            color = self.colors[obj_id]
            # Draw trajectory line
            pygame.draw.lines(self.screen, color, False, trajectory, 3)
            
            # Draw points
            for i, point in enumerate(trajectory):
                radius = 4 if i == 0 else 3  # Larger radius for start point
                pygame.draw.circle(self.screen, color, point, radius)
                
                # Draw point numbers
                if i % 5 == 0:  # Show every 5th point number
                    text = self.small_font.render(str(i), True, self.text_color)
                    self.screen.blit(text, (point[0] + 5, point[1] - 5))
    
    def draw_ui(self):
        """Draw the user interface."""
        # Background for UI - increased height to accommodate all text
        ui_height = 80
        ui_rect = pygame.Rect(0, 0, self.width, ui_height)
        pygame.draw.rect(self.screen, (220, 220, 220), ui_rect)
        pygame.draw.line(self.screen, (180, 180, 180), (0, ui_height), (self.width, ui_height), 2)
        
        # Current object indicator
        current_color = self.colors[self.current_object]
        pygame.draw.circle(self.screen, current_color, (20, 20), 15)
        pygame.draw.circle(self.screen, (0, 0, 0), (20, 20), 15, 2)
        
        # Object name
        obj_text = self.font.render(f"Drawing: {self.object_names[self.current_object]}", True, self.text_color)
        self.screen.blit(obj_text, (50, 15))
        
        # Instructions - reorganized in a single row with better spacing
        instructions = [
            "1-4: Switch objects",
            "S: Save",
            "C: Clear", 
            "R: Reset",
            "ESC: Exit"
        ]
        
        x_offset = 300
        instruction_spacing = 120  # Increased spacing between instructions
        for i, instruction in enumerate(instructions):
            text = self.small_font.render(instruction, True, self.text_color)
            self.screen.blit(text, (x_offset + i * instruction_spacing, 15))
        
        # Trajectory counts - moved to second row with better spacing
        y_offset = 45
        for obj_id in range(4):
            count = len(self.trajectories[obj_id])
            color = self.colors[obj_id]
            text = self.small_font.render(f"{self.object_names[obj_id]}: {count} points", True, color)
            # Use 4 columns instead of 2 to prevent overlap
            col = obj_id % 4
            row = obj_id // 4
            x_pos = 50 + col * 180  # Increased spacing between columns
            y_pos = y_offset + row * 20
            self.screen.blit(text, (x_pos, y_pos))
    
    def add_point(self, pos: Tuple[int, int]):
        """Add a point to the current object's trajectory.
        
        Args:
            pos: (x, y) position in screen coordinates
        """
        self.trajectories[self.current_object].append(pos)
    
    def remove_last_point(self):
        """Remove the last point from the current object's trajectory."""
        if self.trajectories[self.current_object]:
            self.trajectories[self.current_object].pop()
    
    def clear_trajectories(self):
        """Clear all trajectories."""
        self.trajectories = {i: [] for i in range(4)}
    
    def reset_trajectories(self):
        """Reset to start (clear all trajectories and reset object)."""
        self.clear_trajectories()
        self.current_object = 0
    
    def save_trajectories(self, filename: str = None):
        """Save trajectories to a JSON file.
        
        Args:
            filename: Output filename. If None, generates timestamp-based name.
        
        Returns:
            str: The filename used for saving
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trajectory_{timestamp}.json"
        
        # Convert trajectories to the format expected by the model
        # Filter out empty trajectories and convert to the expected format
        data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "canvas_size": [self.width, self.height],
                "num_objects": 4
            },
            "trajectories": {}
        }
        
        for obj_id, trajectory in self.trajectories.items():
            if trajectory:  # Only include non-empty trajectories
                # Convert to the format expected by the model
                # Each trajectory is a list of [x, y] coordinates
                data["trajectories"][f"object_{obj_id}"] = {
                    "positions": trajectory,
                    "color": self.colors[obj_id],
                    "name": self.object_names[obj_id]
                }
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Trajectories saved to: {filename}")
        return filename
    
    def load_trajectories(self, filename: str):
        """Load trajectories from a JSON file.
        
        Args:
            filename: Input filename
        """
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            # Clear existing trajectories
            self.clear_trajectories()
            
            # Load trajectories
            for obj_key, obj_data in data["trajectories"].items():
                obj_id = int(obj_key.split("_")[1])
                self.trajectories[obj_id] = obj_data["positions"]
            
            print(f"Trajectories loaded from: {filename}")
            
        except Exception as e:
            print(f"Error loading trajectories: {e}")
    
    def run(self):
        """Main event loop."""
        clock = pygame.time.Clock()
        running = True
        
        print("Trajectory Drawer Started!")
        print("Controls:")
        print("  - Left mouse: Draw")
        print("  - Right mouse: Erase last point")
        print("  - 1-4: Switch objects")
        print("  - S: Save")
        print("  - C: Clear")
        print("  - R: Reset")
        print("  - ESC: Exit")
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_s:
                        filename = input("Enter filename (or press Enter for auto-generated): ").strip()
                        if not filename:
                            filename = None
                        self.save_trajectories(filename)
                    elif event.key == pygame.K_c:
                        self.clear_trajectories()
                    elif event.key == pygame.K_r:
                        self.reset_trajectories()
                    elif event.key in [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4]:
                        self.current_object = event.key - pygame.K_1
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left mouse button
                        # Only draw if mouse is below the UI area
                        if event.pos[1] > 80:  # UI height
                            self.drawing = True
                            self.add_point(event.pos)
                    elif event.button == 3:  # Right mouse button
                        self.remove_last_point()
                
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        self.drawing = False
                
                elif event.type == pygame.MOUSEMOTION:
                    if self.drawing and event.pos[1] > 80:  # Only draw if mouse is below UI
                        self.add_point(event.pos)
            
            # Draw everything
            self.screen.fill(self.bg_color)
            self.draw_grid()
            self.draw_trajectories()
            self.draw_ui()
            
            pygame.display.flip()
            clock.tick(60)
        
        pygame.quit()


def main():
    """Main function to run the trajectory drawer."""
    if len(sys.argv) > 1:
        print("Usage: python trajectory_drawer.py")
        print("No arguments needed. The tool will start interactively.")
        return
    
    drawer = TrajectoryDrawer()
    drawer.run()


if __name__ == "__main__":
    main()
