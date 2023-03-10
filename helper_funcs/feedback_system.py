class NegativeFeedback:
    def __init__(self, input):
        self.data = input

    def process(self):
        """
        This function will take in the array of mistakes and will follow a hard coded tree structure with all possible
        outcomes of the analysis and will suggest a drill to practice.

        This is a void function. It does the alterations in memory to the input data
        :param input:
        :return:
        """

        """
        check = {
            "Check": "Trail Elbow in Right Direction",
            "Stage": "Backswing",
            "Problem": "Consistency",
            "Description": "",  # Description of what's being done in the swing
            "Fix": "",  # Filled in by the advice feedback system
            "Points": [],
            # all these below used as metadata
            "isMistake": False,
            "isRootCause": False,  # if this mistake leads to another we will need to check for the others
            "LeadsTo": [],
            "isProcessed": False  # once check has been done, mark as true
        }
        """
        print("Processing the results from the analysis to suggest improvements")
        res = []

        for item in self.data:
            if item["isMistake"] and not item["isProcessed"]:
                drill = ""
                if not item["isRootCause"]:
                    item["Fix"] = self.process_solo_mistake(item)  # Assign fix to item (in memory)
                    item["isProcessed"] = True
                else:
                    item["Fix"] = self.process_recursive(item)
                    item["isProcessed"] = True

                # else:
                #     # for r in item["LeadsTo"]:
                #     print("t")
                #     # TODO: Check if m is in input and advise
                # item["fix"] = drill  # fill in the field for the input data

    def process_solo_mistake(self, item):
        """
        This function will process all the solo mistakes (i.e. no recursion needed) and will return a drill.
        :param input:
        :return:
        """
        # print("Stage: " + item["Stage"])
        # print("Preforming check for: " + item["Check"])
        if item["Stage"] == "Setup":
            res = self.process_solo_setup(item)
        elif item["Stage"] == "Takeaway":
            res = self.process_solo_takeaway(item)
        elif item["Stage"] == "Backswing":
            res = self.process_solo_backswing(item)
        elif item["Stage"] == "Downswing":
            res = self.process_solo_downswing(item)
        # elif item["Stage"] == "Impact":
        #     res = self.process_solo_impact(item)
        elif item["Stage"] == "Followthrough":
            res = self.process_solo_followthrough(item)
        # print("Advice gathered: " + res)
        return res

    def process_solo_setup(self, item):
        """
        This function will process the solo setup mistakes and return the feedback as a string
        :param item: 
        :return: 
        """""
        res = ""
        if item["Check"] == "Legs Shoulder Width Apart":
            res = "Using 2 alignment sticks, place them on the floor directly under your shoulders. This will give the an " \
                  "approximate placement of your legs and get used to going into this position. Focus on keeping " \
                  "your centre of gravity in the same position in every setup "
        elif item["Check"] == "Knee Flex Setup":
            if "Your knee is bent too much" in item["Description"] and "Your knee is directly above your ankle" in \
                    item["Description"]:
                res = "In your setup, try standing slightly taller and putting your focus on your centre of gravity " \
                      "and weight distribution being the same everytime "
            elif "Your knee is not bent enough" in item["Description"] and "Your knee is directly above your ankle" in \
                    item["Description"]:
                res = "In your setup, try standing slightly lower and putting your focus on lowering your centre of " \
                      "gravity and weight distribution being the same everytime "
            elif "Knee angle is setup to maintain balance" in item["Description"] and "Your knee is too far ahead" in \
                    item["Description"]:
                res = "Try imagine there are 2 vertical lines going from ur knee caps to your ankles. These lines " \
                      "should be parallel with each other. You should also focus on trying the bring your centre of " \
                      "gravity back a little "
            elif "Knee angle is setup to maintain balance" in item["Description"] and "Your knee is behind your ankle" \
                    in item["Description"]:
                res = "Try imagine there are 2 vertical lines going from your knee caps to your ankles. These lines " \
                      "should be parallel with each other. You should also focus on trying the bring your centre of " \
                      "gravity forward a little "

        return res

    def process_solo_takeaway(self, item):
        """
        This function will process the mistakes that occurred in the takeaway
        :param item:
        :return:
        """
        res = ""
        if item["Check"] == "Legs Shoulder Width Apart":
            res = "Try doing the pause drill. In this drill you will setup as normal and you would lift the swing the " \
                  "club as you normally would except once the club reaches waist height you will pause and make sure " \
                  "your shoulders, elbows and wrists are all inline moving as one. Repeat this until it becomes habit "
        elif item["Check"] == 'Trail Arm Straight':
            res = "Try using the straight arm drill. Setup as you would for a regular shot but before you start " \
                  "swinging the club maike sure your trail arm is extended fully and is completely straight. Maintain " \
                  "this stretch for as long as you can until the natural bend of arm comes into play "
        elif item["Check"] == 'One Piece Movement':
            res = "Try using the straight arm drill. Setup as you would for a regular shot but before you start " \
                  "swinging the club maike sure your trail arm is extended fully and is completely straight. Maintain " \
                  "this stretch for as long as you can until the natural bend of arm comes into play "
        return res

    def process_solo_backswing(self, item):
        """
        This function will process the mistakes that occurred in the backswing
        :param item:
        :return:
        """
        res = ""
        if item["Check"] == "Lead Arm on Shoulder Plane":
            res = "Place an alignment stick on the ground facing in the same direction as your shoulders are. Then " \
                  "swing as normal but pause at the backswing and and verify that your lead arm is on the same plane " \
                  "as your shoulders, adjusting your swing axis as necessary. Do this until you are arm is " \
                  "consistently on the same plane and then start using it to hit your shot "
        elif item["Check"] == "Trail Elbow in Right Direction":
            res = "Try preforming the towel drill. When you setup place a towel under your trail armpit. Now when you " \
                  "swing you should try to keep this towel in place under your armpit for the duration of the " \
                  "backswing. Repeat this until you are doing it subconsciously "
        return res

    def process_solo_downswing(self, item):
        """
        This function will process the mistakes that occurred in the downswing
        :param item:
        :return:
        """
        res = ""
        if item["Check"] == "Head Behind Ball":
            res = "Place an alignment stick perpendicular to your stance and directly to the middle of the ball. When " \
                  "you swing, make sure to keep your head slightly behind the alignment stick. If required slow " \
                  "down a little bit and just focus on keeping your head behind the stick. "
        elif item["Check"] == "Knee Flex":
            res = "Place 2 small sticks vertical from your hell to your knee on either leg. Now when you swing make " \
                  "sure to maintain contact with this stick at all times especially around your knee. "
        elif item["Check"] == "Closed Shoulders":
            if "Your shoulders were open" in item["Description"]:
                res = "Try preforming the towel drill. When you setup place a towel under your lead armpit. Now when " \
                      "you swing you should try to keep this towel in place under your armpit for the duration of the " \
                      "downswing. Repeat this until you are doing it subconsciously "
            else:
                res = "Stick an alignment stick along your shoulders in such a way that in setup it is facing in the " \
                      "direction of your target. Now slowly swing up to your backswing and thn as you beguin your " \
                      "downswing focus on the allignment stick pointing slightly left of the ball and gradually " \
                      "getting larger. Repeat this and increase the speed as you get better untill you can start " \
                      "hitting the ball while doing this drill. "
        return res

    def process_solo_followthrough(self, item):
        """
        This function will process the mistakes that occurred in the followthrough
        :param item:
        :return:
        """
        res = ""
        if item["Check"] == "Head over Lead Leg":
            res = "Place an alignment stick along your shoulders. Hold the stick in place with both your hands. Using " \
                  "your body (no hand movement) rotate your body as it would during a swing. When the followership " \
                  "approaches focus on shifting your centre of gravity forward slightly in order to push your head " \
                  "over directly above your foot. "
        return res

    def process_recursive(self, item):
        """
        This function will carry out the recursion between branches. Basically takes the input array and item in as
        input. Will check if item need to recur. If so it will check that the "LeadsTo" field is recognised in the
        video and if so it will check for this. Otherwise it will continue as normal
        :param item:
        :return:
        """
        # This is how we assign a value to the data - item["Points"].append([245, 255])
        # Make a copy of current item just in case
        temp = item
        original_index = [index for index in range(len(self.data)) if self.data[index] == temp]
        mistake_titles = [item["Check"] for item in self.data if item["isMistake"]]

        for title in mistake_titles.copy():
            if title in item["LeadsTo"]:
                print("{} brings on {} seen in the swing.".format(item["Check"], title))

                index = [index for index in range(len(self.data)) if self.data[index]["Check"] == title]

                res = "The drill for {}, but may help to alleviate from {}.".format(item["Check"], title)

                self.data[index[0]]["Fix"] = res
                self.data[index[0]]["isProcessed"] = True

            # else:
            #     continue  # Next loop as we did not see this problem in
            mistake_titles.remove(title)
        return self.process_solo_mistake(item)
