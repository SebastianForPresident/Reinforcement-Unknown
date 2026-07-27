from pynput import keyboard

_server = None

def Start(server):
    global _server, _listener
    _server = server

    _listener = keyboard.Listener(
        on_press=on_press,
        on_release=on_release
    )
    _listener.start()

def clampdX(v):
    return max(-4, min(4, v))

def clampdY(v):
    return max(-5, min(5, v))

def clampBag(v):
    return max(-1, v)

def clampIndex(v):
    return max(0, min(24, v))

def clampSelIndex(v):
    return max(-1, min(24, v))

def clampLimb(v):
    return max(0, min(14, v))

def clampRecipe(v):
    return max(-1, min(131, v))

def clampExercise(v):
    return max (-1, min(2, v))

def clampLiquid(v):
    return max (0, min(1000, v))

def on_press(key):
    try:
        specialKey = None
        if (isinstance(key, keyboard.Key)):
            specialKey = key
            key = keyboard.KeyCode(char=None)

        if key.char in ['h', 'c'] or specialKey == keyboard.Key.tab: # mode switching takes priority
            if specialKey == keyboard.Key.tab:
                if _server.mode == "inventory":
                    _server.mode = "none"
                else:
                    _server.mode = "inventory"

            elif key.char == 'h':
                if _server.mode == "medical":
                    _server.mode = "none"
                else:
                    _server.mode = "medical"

            elif key.char == 'c':
                if _server.mode == "craft":
                    _server.mode = "none"
                else:
                    _server.mode = "craft"
        else:
            if _server.mode == "none":
                # Basic locomotion
                if key.char == 'a':
                    _server.move = -1
                elif key.char == 'd':
                    _server.move = 1

                if key.char == "w":
                    _server.vertMove = 1
                elif key.char == "s":
                    _server.vertMove = -1

                if specialKey == keyboard.Key.space:
                    _server.jump = 1

                if key.char == "s":
                    _server.crouch = 1

                if key.char == 'x':
                    _server.ragdoll = 1

                # Cursor
                if specialKey == keyboard.Key.left:
                    _server.lookdX -= 1
                elif specialKey == keyboard.Key.right:
                    _server.lookdX += 1

                if specialKey == keyboard.Key.up:
                    _server.lookdY += 1
                elif specialKey == keyboard.Key.down:
                    _server.lookdY -= 1

                # Actions
                if key.char == 'f':
                    _server.attack = 1

                if key.char == 'e':
                    _server.interact = 1

                if key.char == 't':
                    _server.throw = 1

                if key.char == 'r':
                    _server.useItemWorld = 1

                if key.char == 'b':
                    _server.bark = 1

                if key.char == "z":
                    _server.pullLiquidFromWorld = 1
        
            if _server.mode == "inventory":
                # Indexing
                if key.char == 'w':
                    _server.targetSlotIndex += 1
                if key.char == 's':
                    _server.targetSlotIndex -= 1

                if key.char == 'a':
                    _server.selectedSlotIndex -= 1
                if key.char == 'd':
                    _server.selectedSlotIndex += 1

                if key.char == 'q':
                    _server.selectedBagIndex = clampBag(_server.selectedBagIndex - 1)
                if key.char == 'e':
                    _server.selectedBagIndex += 1

                if key.char == 'v':
                    _server.liquidAmount -= 5
                if key.char == 'b':
                    _server.liquidAmount += 5

                # Actions
                if key.char == 't':
                    _server.moveItem = 1

                if key.char == 'f':
                    _server.favoriteItem = 1

                if key.char == 'r':
                    _server.useItem = 1

                if key.char == 'x':
                    _server.dropItem = 1

                if key.char == 'z':
                    _server.drainLiquid = 1
            
            if _server.mode == "medical":
                # Indexing
                if key.char == 'a':
                    _server.selectedLimb -= 1
                if key.char == 'd':
                    _server.selectedLimb += 1

                if key.char == 'w':
                    _server.selectedSlotIndex += 1
                if key.char == 's':
                    _server.selectedSlotIndex -= 1

                # Actions
                if key.char == 'e':
                    _server.useItemMedical = 1

                if key.char == 'q':
                    _server.switchMainHand = 1

                if key.char == 'r':
                    _server.trySleep = 1

                if key.char == '1':
                    _server.exercise = 0
                if key.char == '2':
                    _server.exercise = 1
                if key.char == '3':
                    _server.exercise = 2

            if _server.mode == "craft":
                # Indexing
                if key.char == 'a':
                    _server.chosenRecipe = clampRecipe(_server.chosenRecipe - 1)
                if key.char == 'd':
                    _server.chosenRecipe = clampRecipe(_server.chosenRecipe + 1)
                
                # Actions
                if key.char == 'w':
                    _server.selectedRecipe = _server.chosenRecipe

        _server.lookdX = clampdX(_server.lookdX)
        _server.lookdY = clampdY(_server.lookdY)
        _server.targetSlotIndex = clampIndex(_server.targetSlotIndex)
        _server.selectedSlotIndex = clampSelIndex(_server.selectedSlotIndex)
        _server.selectedLimb = clampLimb(_server.selectedLimb)
        _server.selectedRecipe = clampRecipe(_server.selectedRecipe)
        _server.exercise = clampExercise(_server.exercise)
        _server.liquidAmount = clampLiquid(_server.liquidAmount)

    except AttributeError:
        pass

def on_release(key):
    try:
        specialKey = None
        if (isinstance(key, keyboard.Key)):
            specialKey = key
            key = keyboard.KeyCode(char=None)


        if _server.mode == "none":
            # Basic locomotion
            if key.char in ['a','d']:
                _server.move = 0
            if key.char in ['w','s']:
                _server.vertMove = 0

            if specialKey == keyboard.Key.space:
                _server.jump = 0

            if key.char == 's':
                _server.crouch = 0

            if key.char == 'x': # need to try with movement later, also move this to locomotion and promote it from action
                _server.ragdoll = 0

            # Actions
            if key.char == 'f':
                _server.attack = 0

            if key.char == 'e':
                _server.interact = 0

            if key.char == 't':
                _server.throw = 0

            if key.char == 'r':
                _server.useItemWorld = 0

            if key.char == 'b':
                _server.bark = 0

            if key.char == "z":
                _server.pullLiquidFromWorld = 0
    
        if _server.mode == "inventory":
            # Actions
            if key.char == 't':
                _server.moveItem = 0

            if key.char == 'f':
                _server.favoriteItem = 0

            if key.char == 'r':
                _server.useItem = 0

            if key.char == 'x':
                _server.dropItem = 0

            if key.char == 'z':
                _server.drainLiquid = 0
        
        if _server.mode == "medical":
            # Actions
            if key.char == 'e':
                _server.useItemMedical = 0

            if key.char == 'q':
                _server.switchMainHand = 0

            if key.char == 'r':
                _server.trySleep = 0

            if key.char in ['1','2','3']:
                _server.exercise = -1
            
        if _server.mode == "craft":    
            # Actions
            if key.char == 'w':
                _server.selectedRecipe = -1
        

    except AttributeError:
        pass
