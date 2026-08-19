import time

from src.leika import Robot, Gait

def main():
    print("--- Spot Micro Leika: Visionary API Demo ---")
    
    # Use the 'with' context manager for automatic safety/cleanup
    with Robot(target="simulation") as spot:
        
        # 1. Stand up
        print("\n[*] Standing up...")
        spot.stand(height=0.8)
        time.sleep(2)
        
        # 2. Show off some "Dance" moves (Kinematics)
        print("[*] Dancing...")
        for i in range(3):
            spot.walk(turn=0.5) # Leaning
            time.sleep(0.5)
            spot.walk(turn=-0.5)
            time.sleep(0.5)
        
        spot.stand(height=0.5)
        time.sleep(1)

        # 3. Walk forward
        print("[*] Walking forward...")
        spot.walk(x=0.8, speed=1.5)
        
        # Monitor position
        for _ in range(30):
            pos = spot.position
            print(f"    Current Pos: x={pos[0]:.2f}, y={pos[1]:.2f}, z={pos[2]:.2f}", end="\r")
            time.sleep(0.1)
        
        print("\n[*] Stopping...")
        spot.stand()
        time.sleep(2)

        # 4. Rest (Sits down)
        print("[*] Resting...")
        spot.rest()
        time.sleep(2)

    print("\n--- Demo Complete ---")

if __name__ == "__main__":
    main()
