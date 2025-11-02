from abc import ABC, abstractmethod

class IBotService(ABC):

    @abstractmethod
    def shouldActive(self, msg: str):
        pass

    @abstractmethod
    def execute(self, msg: str):
        pass

class ChatRoom:

    def __init__(self):
        self.global_state = {}
        self.bots = []
    
    def register_bot(self, bot: IBotService):
        self.bots.append(bot)

    def on_message(self, msg: str) -> str:
        for bot in self.bots:
            if bot.shouldActive(msg):
                return bot.execute(msg)
        return "no bot matched"

class MeetingBot(IBotService):

    def __init__(self):
        pass
    
    def shouldActive(self, msg: str):
        if "meeting" in msg:
            return True

        return False

    def execute(self, msg: str):
        return f"{msg}: meeting is scheduled at abc"

class TacoBot(IBotService):

    def __init__(self, global_state: dict):
        self.global_state = global_state

    def shouldActive(self, msg: str):
        return "taco" in msg
    
    def execute(self, msg: str):
        if "taco" not in self.global_state:
            self.global_state["taco"] = 1
        else:
            self.global_state["taco"] += 1
        return f"In total, {self.global_state['taco']} is ordered"


chat_room = ChatRoom()
meeting_bot = MeetingBot()
chat_room.register_bot(meeting_bot)
taco_bot = TacoBot(chat_room.global_state)
chat_room.register_bot(taco_bot)

print(chat_room.on_message("hello"))
print(chat_room.on_message("meeting test test"))
print(chat_room.on_message("taco"))
print(chat_room.on_message("another taco"))