import cv2
import numpy as np

TILE_SIZE = 16
LOOK_RANGE_X = 42
LOOK_RANGE_Y = 24

FULL_FOV_X = LOOK_RANGE_X*2+1
FULL_FOV_Y = LOOK_RANGE_Y*2+1

STAT_X = 10
STAT_Y = FULL_FOV_Y*TILE_SIZE + 20

COLUMN_WIDTH = 170
LINE_HEIGHT = 18
ROWS_PER_COLUMN = 14

SIDE_PANEL_X = FULL_FOV_X * TILE_SIZE + 10
SIDE_PANEL_Y = 20
SIDE_LINE = 18

stat_index = 0

def Stat(text, color=(255,255,255)):
    global stat_index

    col = stat_index // ROWS_PER_COLUMN
    row = stat_index % ROWS_PER_COLUMN

    x = STAT_X + col * COLUMN_WIDTH
    y = STAT_Y + row * LINE_HEIGHT

    cv2.putText(
        window,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA
    )

    stat_index += 1

def SideText(text, line, color=(255,255,255)):
    cv2.putText(
        window,
        text,
        (SIDE_PANEL_X, SIDE_PANEL_Y + line * SIDE_LINE),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA
    )

window = np.zeros((FULL_FOV_Y*TILE_SIZE + 260,FULL_FOV_X*TILE_SIZE + 300,3), dtype=np.uint8) # extra constants for stats info

def Update(obs, aux):
    global window
    global stat_index

    window[:] = 0

    # Relative Tile Map
    for y in range(FULL_FOV_Y): # Y sight range is 48 + center block
        for x in range(FULL_FOV_X): # X sight range is 84 + center block
            block = obs["RelativeBlockMap"][x][y]

            brightness = int(block["Health"] * 255 / 100)

            cv2.rectangle(
                window,
                (x*TILE_SIZE, y*TILE_SIZE),
                ((x+1)*TILE_SIZE, (y+1)*TILE_SIZE),
                (brightness, brightness, brightness),
                -1
            )

    # Relative Fluid Map
    for y in range(FULL_FOV_Y): # Y sight range is 10 + center block
        for x in range(FULL_FOV_X): # X sight range is 16 + center block
            fluid = obs["RelativeFluidMap"][x][y]
            color = (0,0,255) # obvious red in case something explodes

            match fluid["Type"]:
                case 1:
                    color = (255,0,0)
                case 2:
                    color = (0,255,0)

            if fluid["Type"] != 0: # not air
                cv2.rectangle(window,(x*TILE_SIZE, y*TILE_SIZE),((x+1)*TILE_SIZE, (y+1)*TILE_SIZE),color,-1)
    
    # Visible Buildings
    try:
        for i in range(len(obs["VisibleBuildings"])):
            if obs["VisibleBuildings"][i]["Exists"]:
                building = obs["VisibleBuildings"][i]

                xPos = (LOOK_RANGE_X + building["RelativePosition"]["X"]) * TILE_SIZE + TILE_SIZE // 2 
                yPos = (LOOK_RANGE_Y - building["RelativePosition"]["Y"]) * TILE_SIZE + TILE_SIZE // 2 # OpenCV flipping vertical for some reason

                cv2.circle(window, (xPos, yPos), 5, (0,255,255), -1)
    except Exception as e:
        print(e)

    # Visible Items
    try:
        for i in range(len(obs["VisibleItems"])):
            if obs["VisibleItems"][i]["Item"]["ID"] >= 0:
                item = obs["VisibleItems"][i]

                xPos = (LOOK_RANGE_X + item["RelativePosition"]["X"]) * TILE_SIZE + TILE_SIZE // 2 
                yPos = (LOOK_RANGE_Y - item["RelativePosition"]["Y"]) * TILE_SIZE + TILE_SIZE // 2 # OpenCV flipping vertical for some reason

                cv2.circle(window, (xPos, yPos), 5, (0,255,0), -1)
    except Exception as e:
        print(e)

    # Timmy
    cv2.circle(window, (LOOK_RANGE_X*TILE_SIZE + TILE_SIZE//2, LOOK_RANGE_Y*TILE_SIZE + TILE_SIZE//2), 5, (0,165,255), -1)

    # Relative Cursor
    cv2.circle(window, ((LOOK_RANGE_X + obs["RelativeLookPos"]["X"])*TILE_SIZE + TILE_SIZE//2, (LOOK_RANGE_Y - obs["RelativeLookPos"]["Y"])*TILE_SIZE + TILE_SIZE//2), 3, (0,0,255), -1)

    # Separators
    cv2.line(
        window,
        (FULL_FOV_X * TILE_SIZE, 0),
        (FULL_FOV_X * TILE_SIZE, window.shape[0]),
        (80, 80, 80),
        1
    )

    cv2.line(
        window,
        (0, FULL_FOV_Y * TILE_SIZE),
        (window.shape[1], FULL_FOV_Y * TILE_SIZE),
        (80, 80, 80),
        1
    )

    # STATISTICS
    stat_index = 0

    # Basic Locomotion
    Stat(f"VelX: {obs['Velocity']['X']:.2f}")
    Stat(f"VelY: {obs['Velocity']['Y']:.2f}")
    Stat(f"Right: {obs['IsRight']}")
    Stat(f"MaxSpd: {obs['MaxSpeed']:.2f}")

    # Jumping
    Stat(f"JumpCD: {obs['JumpCooldown']:.2f}")
    Stat(f"Ground: {obs['Grounded']}")
    Stat(f"SinceG: {obs['TimeSinceGrounded']:.2f}")

    # Ragdolling
    Stat(f"Ragdoll: {obs['TimeRagdolled']:.2f}")
    Stat(f"Crawl: {obs['CrawlTime']:.2f}")

    # Water
    Stat(f"InWater: {obs['InWater']}")
    Stat(f"Slip: {obs['LiquidSlipTime']:.2f}")
    Stat(f"LiqRag: {obs['LiquidRagdollBar']:.2f}")
    Stat(f"Drink: {obs['LiquidDrinkTime']:.2f}")

    # Walljump
    Stat(f"WallL: {obs['CanWalljumpLeft']}")
    Stat(f"WallR: {obs['CanWalljumpRight']}")

    # Combat
    Stat(f"AtkCD: {obs['AttackCooldown']:.2f}")

    # Crouching
    Stat(f"Crouch: {obs['CrouchAmount']:.2f}")
    Stat(f"Crouching: {obs['Crouching']}")

    # Climbing
    Stat(f"Climb: {obs['IsClimbing']}")
    Stat(f"ClimbP: {obs['ClimbableProgress']:.2f}")
    Stat(f"ClimbV: {obs['ClimbVelocity']:.2f}")

    # Heart
    Stat(f"Heart: {obs['HeartRate']:.1f}")
    Stat(f"Fib: {obs['FibrillationProgress']:.2f}")
    Stat(f"FibForce: {obs['FibrillationForced']}")
    Stat(f"FibRise: {obs['FibrillationRising']}")
    Stat(f"PE: {obs['HasPulmonaryEmbolism']}")

    # Blood
    Stat(f"O2: {obs['BloodOxygen']:.1f}")
    Stat(f"Blood: {obs['BloodVolume']:.1f}")
    Stat(f"BP: {obs['BloodPressure']:.1f}")
    Stat(f"Vessel: {obs['BloodVesselSize']:.2f}")
    Stat(f"Visc: {obs['BloodViscosity']:.2f}")
    Stat(f"Bleed: {obs['TotalBleedSpeed']:.2f}")
    Stat(f"IBleed: {obs['InternalBleeding']:.2f}")
    Stat(f"Hemo: {obs['Hemothorax']:.2f}")
    Stat(f"VenTot: {obs['VenomTotal']:.2f}")
    Stat(f"VenCur: {obs['VenomCurrent']:.2f}")

    # Breathing
    Stat(f"Resp: {obs['RespiratoryRate']:.1f}")
    Stat(f"Breath: {obs['Breathing']}")

    # Adrenaline
    Stat(f"Adr: {obs['Adrenaline']:.1f}")
    Stat(f"CurAdr: {obs['CurAdrenaline']:.1f}")

    # Drugs
    Stat(f"Stim: {obs['StimulantMultiplier']:.2f}")
    Stat(f"HardStim: {obs['OnHardStimulants']}")
    Stat(f"Opiate: {obs['OpiateHappiness']:.1f}")
    Stat(f"AD: {obs['AntidepressantHappiness']:.1f}")
    Stat(f"BrainGrow: {obs['BrainGrowSickness']:.1f}")
    Stat(f"Neural: {obs['UsedNeuralBooster']}")
    Stat(f"MindWipe: {obs['MindWiped']}")
    Stat(f"Caff: {obs['Caffeinated']:.1f}")
    Stat(f"OD: {obs['OverdoseIndex']}")

    # Misc
    Stat(f"Weight: {obs['WeightOffset']:.1f}")
    Stat(f"Hunger: {obs['Hunger']:.1f}")
    Stat(f"Thirst: {obs['Thirst']:.1f}")
    Stat(f"Stam: {obs['Stamina']:.1f}")
    Stat(f"Energy: {obs['Energy']:.1f}")
    Stat(f"Immune: {obs['Immunity']:.1f}")
    Stat(f"Happy: {obs['TotalHappiness']:.1f}")
    Stat(f"Dirty: {obs['Dirtyness']:.1f}")
    Stat(f"Claw: {obs['ClawHealth']:.1f}")

    # Brain
    Stat(f"Brain: {obs['BrainHealth']:.1f}")
    Stat(f"Con: {obs['Consciousness']:.1f}")
    Stat(f"Shock: {obs['Shock']:.1f}")
    Stat(f"RevCtrl: {obs['ReversedControls']}")
    Stat(f"Dying: {obs['BrainDying']}")
    Stat(f"Stroke: {obs['StrokeAmount']:.2f}")

    # Temperature
    Stat(f"Temp: {obs['Temperature']:.1f}")
    Stat(f"Clothes: {obs['ClothingTemperature']:.1f}")

    # Pain
    Stat(f"Pain: {obs['AveragePain']:.1f}")
    Stat(f"PShock: {obs['PainShock']:.1f}")
    Stat(f"Hearing: {obs['HearingLoss']:.1f}")
    Stat(f"NoHands: {obs['BothHandsUnusable']}")

    # Disease
    Stat(f"Sick: {obs['SicknessAmount']:.1f}")
    Stat(f"Septic: {obs['SepticShock']:.1f}")
    Stat(f"Rad: {obs['RadiationSickness']:.1f}")

    # Mental
    Stat(f"Corpses: {obs['CorpsesSeen']}")
    Stat(f"Trauma: {obs['TraumaAmount']:.1f}")
    Stat(f"Horror: {obs['HorrifiedLevel']:.1f}")
    Stat(f"Focus: {obs['FocusedLevel']:.1f}")

    # Dismemberment
    Stat(f"Disfig: {obs['Disfigured']}")
    Stat(f"EyeGone: {obs['EyeGone']}")
    Stat(f"Blind: {obs['BothEyesGone']}")

    # Encumbrance
    Stat(f"Enc: {obs['TotalEncumberance']:.1f}")
    Stat(f"OverEnc: {obs['OverEncumberance']:.1f}")
    Stat(f"MaxEnc: {obs['MaxEncumberance']:.1f}")

    # Sleep
    Stat(f"Sleep: {obs['Sleeping']}")
    Stat(f"SleepQ: {obs['CurSleep']}")
    Stat(f"BadSleep: {obs['BadSleepAmount']:.1f}")
    Stat(f"GoodSleep: {obs['GoodSleepTime']:.1f}")
    Stat(f"ForcedQ: {obs['ForcedSleepQuality']}")
    Stat(f"Bag: {obs['UsingSleepingBag']}")
    Stat(f"Nap: {obs['CanTakeNap']}")

    # Last Stand
    Stat(f"TriedLS: {obs['TriedRollingLastStand']}")
    Stat(f"LSTime: {obs['LastStandTime']:.2f}")

    # Skills
    Stat(f"STR: {obs['STR']}")
    Stat(f"RES: {obs['RES']}")
    Stat(f"INT: {obs['INT']}")
    Stat(f"STRXP: {obs['STRProgress']:.2f}")
    Stat(f"RESXP: {obs['RESProgress']:.2f}")
    Stat(f"INTXP: {obs['INTProgress']:.2f}")

    # Progress
    Stat(f"Layer: {obs['CurrentLayer']}")
    Stat(f"LayerP: {obs['LayerProgress']:.2f}")
    Stat(f"BestDep: {obs['BestLayerDepth']}")

    # Radline
    Stat(f"TimeLeft: {obs['LayerTimeRemaining']}")
    Stat(f"RadDisp: {obs['RadLineDisplacement']}")

    # AUX + SELECTED INFO
    def RelativeSideText(display):
        nonlocal line
        SideText(display, line)
        line += 1

    def DisplayItem(item):
        RelativeSideText(f"ID: {item['ID']}")
        RelativeSideText(f"Cond: {item['Condition']:.2f}")

        for quality in item["Qualities"]:
            if quality["ID"] != -1:
                RelativeSideText(f"Q{quality['ID']}: {quality['Amount']:.1f}")

        for liquid in item["Liquids"]:
                if liquid["ID"] != -1:
                    RelativeSideText(f"L{liquid['ID']}: {liquid['Amount']}")

                    for liquidQuality in liquid["Qualities"]:
                        if liquidQuality["ID"] != -1:
                            RelativeSideText(f"Q{liquidQuality['ID']}: {liquidQuality['Amount']:.1f}")
        
    def DisplayBuilding(building):
        RelativeSideText(f"Health: {building['Health']:.2f}")
        RelativeSideText(f"Pos: ({building['RelativePosition']['X']}, {building['RelativePosition']['Y']})")

        for drop in building["DropPool"]:
            if drop != -1:
                RelativeSideText(f"Drop: {drop}")

    # None
    if aux["Mode"] == "none":
        line = 0
        
        RelativeSideText("=== Main ===")

        RelativeSideText(f"LiquidAmount: {aux['LiquidAmount']}")

        line += 1

        RelativeSideText(f"Main Hand:")

        DisplayItem(obs["Inventory"][0])

        line += 1

        RelativeSideText(f"Secondary Hand:")

        DisplayItem(obs["Inventory"][1])

        line += 1

        RelativeSideText(f"Target:")
        RelativeSideText(f"Slot: {aux['TargetSlot']}")
        DisplayItem(obs["Inventory"][aux["TargetSlot"]])

    if aux["Mode"] == "inventory":
        line = 0
        
        RelativeSideText("=== Inventory ===")

        RelativeSideText(f"Bag Index: {aux['SelectedBagIndex']}")
        RelativeSideText(f"LiquidAmount: {aux['LiquidAmount']}")

        line += 1

        RelativeSideText(f"Selected:")
        RelativeSideText(f"Slot: {aux['SelectedSlot']}")

        DisplayItem(obs["Inventory"][aux["SelectedSlot"]])
        
        if aux["SelectedBagIndex"] >= 0:
            try:
                line += 1
                RelativeSideText(f"--- Selected Bag Item ---")
                DisplayItem(obs["Inventory"][aux["SelectedSlot"]]["Contents"][aux["SelectedBagIndex"]])
            except Exception as e:
                print("it's inventory")
                print(e)

        line += 1

        RelativeSideText(f"Target:")
        RelativeSideText(f"Slot: {aux['TargetSlot']}")
        DisplayItem(obs["Inventory"][aux["TargetSlot"]])

    # Recipe
    if aux["Mode"] == "craft":
        SideText("=== Recipe ===", 0)
        if aux["ChosenRecipe"] != -1:
            try:
                recipe = obs["Recipes"][aux["ChosenRecipe"]]

                SideText(f"Index: {aux['ChosenRecipe']}", 1)
                SideText(f"Craftable: {recipe['IsCraftable']}", 2)

                SideText(f"Output ID: {recipe['Output']['ID']}", 3)
                SideText(f"Output Amt: {recipe['OutputQuantity']}", 4)
                SideText(f"Output Cond: {recipe['Output']['Condition']:.0f}", 5)

                SideText("Items:", 7)
                ingredientLine = 8

                for item in recipe["ItemRequirements"]:
                    if item['ID'] != -1:
                        SideText(
                            f"{item['ID']} ({item['Condition']:.0f}%)",
                            ingredientLine
                        )
                        ingredientLine += 1

                ingredientLine += 1
                SideText("Qualities:", ingredientLine)
                ingredientLine += 1
                        
                for quality in recipe["QualityRequirements"]:
                    if quality['ID'] != -1:
                        SideText(
                            f"Q{quality['ID']} ({quality['Amount']:.0f}%)",
                            ingredientLine
                        )
                        ingredientLine += 1
            except Exception as e:
                print("it's crafting")
                print(e)
    
    # Limb
    if aux["Mode"] == "medical":
        limb = obs["Limbs"][aux["SelectedLimb"]]

        SideText("=== Limb ===", 0)
        SideText(f"Index: {aux['SelectedLimb']}", 1)

        SideText(f"Skin: {limb['SkinHealth']:.1f}", 2)
        SideText(f"Muscle: {limb['MuscleHealth']:.1f}", 3)
        SideText(f"Pain: {limb['Pain']:.1f}", 4)
        SideText(f"Infection: {limb['InfectionAmount']:.2f}", 5)
        SideText(f"Disinfect: {limb['DisinfectionTime']:.1f}", 6)

        SideText(f"Dislocated: {limb['Dislocated']}", 7)
        SideText(f"Broken: {limb['Broken']}", 8)
        SideText(f"Splinted: {limb['Splinted']}", 9)
        SideText(f"Infected: {limb['Infected']}", 10)

        SideText(f"Disloc Timer: {limb['DislocationTimer']:.1f}", 11)
        SideText(f"Bone Heal: {limb['BoneHealTimer']:.1f}", 12)

        SideText(f"Vital: {limb['IsVital']}", 13)
        SideText(f"Head: {limb['IsHead']}", 14)
        SideText(f"Abdomen: {limb['IsAbdomen']}", 15)
        SideText(f"Leg: {limb['IsLegLimb']}", 16)
        SideText(f"Arm: {limb['IsArm']}", 17)

        SideText(f"Heart Dist: {limb['DistanceToHeart']}", 18)
        SideText(f"Shrapnel: {limb['Shrapnel']}", 19)

        SideText(f"Dismembered: {limb['Dismembered']}", 20)
        SideText(f"Bleed: {limb['TotalBleedAmount']:.2f}", 21)
    
    # Inspection
    line = 30
    RelativeSideText("--- Inspector ---")
    
    cursorXPos = (LOOK_RANGE_X + obs["RelativeLookPos"]["X"])*TILE_SIZE + TILE_SIZE//2
    cursorYPos = (LOOK_RANGE_Y - obs["RelativeLookPos"]["Y"])*TILE_SIZE + TILE_SIZE//2

    # Visible Buildings
    try:
        for i in range(len(obs["VisibleBuildings"])):
            if obs["VisibleBuildings"][i]["Exists"]:
                building = obs["VisibleBuildings"][i]

                xPos = (LOOK_RANGE_X + building["RelativePosition"]["X"]) * TILE_SIZE + TILE_SIZE // 2 
                yPos = (LOOK_RANGE_Y - building["RelativePosition"]["Y"]) * TILE_SIZE + TILE_SIZE // 2 # OpenCV flipping vertical for some reason
                if xPos == cursorXPos and yPos == cursorYPos:
                    DisplayBuilding(building)
    except Exception as e:
        print(e)

    # Visible Items
    try:
        for i in range(len(obs["VisibleItems"])):
            if obs["VisibleItems"][i]["Item"]["ID"] >= 0:
                item = obs["VisibleItems"][i]

                xPos = (LOOK_RANGE_X + item["RelativePosition"]["X"]) * TILE_SIZE + TILE_SIZE // 2 
                yPos = (LOOK_RANGE_Y - item["RelativePosition"]["Y"]) * TILE_SIZE + TILE_SIZE // 2 # OpenCV flipping vertical for some reason

                if xPos == cursorXPos and yPos == cursorYPos:
                    DisplayItem(item["Item"])
    except Exception as e:
        print(e)

    # Sounds
    SideText("=== Sounds ===", 44)

    line = 45

    for sound in obs["SoundsHeard"]:
        if sound["ID"] == -1:
            continue

        SideText(f"ID: {sound['ID']}", line)
        SideText(f"Pos: ({sound['RelativeTilePosition']['X']}, {sound['RelativeTilePosition']['Y']})", line + 1)
        SideText(f"Vol: {sound['Volume']:.2f}", line + 2)

        line += 4

    cv2.imshow("CasU Vision", window)
    cv2.waitKey(1)
