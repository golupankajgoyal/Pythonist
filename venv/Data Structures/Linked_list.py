class Node :
    def __init__(self,data):
        self.data=data
        self.next=None

class linked_list :

    def __init__(self):
        self.head=None

    def append(self,List):

        if len(List)<=0 :
            return

        new_node = Node(List[0])
        if self.head is None :
            self.head=new_node
        else:
            new_node.next=self.head
            self.head=new_node

        self.append(List[1:])

    def reverse(self):
        prev=None
        curr=self.head

        while curr is not None :
            next=curr.next
            curr.next=prev
            prev=curr
            curr=next
        self.head=prev

def Print(curr):
    print(curr.data)
    while curr.next is None :
        return
    curr=curr.next
    Print(curr)



a=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13,14,15]
llista=linked_list()
llista.append(a)
llista.reverse()
Print(llista.head)
b=[3,7,8]
llistb=linked_list()
llistb.append(b)
llistb.reverse()
Print(llistb.head)

