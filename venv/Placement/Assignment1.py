class Person():
    def __init__(self, name):
        self.name = name

    def talk(self):
        print(f"hii {self.name}")


a = Person("Pankaj Goyal")
a.talk()

