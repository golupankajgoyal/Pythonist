
class Node:
    def __init__(self,value):
        self.value=value
        self.next=None

def printLL(head):
    while head!=None:
        print(head.value,end=" ")
        head=head.next
    print("")

head=Node(1)
head.next=Node(2)
head.next.next=Node(3)
head.next.next.next=Node(4)
head.next.next.next.next=Node(5)
head.next.next.next.next.next=Node(6)
head.next.next.next.next.next.next=Node(7)
head.next.next.next.next.next.next.next=Node(8)

# Given the head of a LinkedList and a number ‘k’, reverse every ‘k’ sized sub-list starting from the head.
def reverseEveryK(head,k):
    current,prev=head,None
    while True:
        last_element_prev_list=prev
        last_element_reverse_list=current
        next=None
        i=0
        while current and i<k:
            next=current.next
            current.next=prev
            prev=current
            current=next
            i+=1

        if last_element_prev_list:
            last_element_prev_list.next=prev
        else:
            head=prev
        last_element_reverse_list.next=current
        if current is None:
            return head
        prev=last_element_reverse_list

# Given the head of a Singly LinkedList and a number ‘k’, rotate the LinkedList to the right by ‘k’ nodes.

def rotateByK(head,rotation):
    if head is None or head.next is None or rotation<1:
        return

    last_node=head
    length=1
    while last_node.next:
        last_node=last_node.next
        length+=1

    last_node.next=head
    rotation%=length
    skip_len=length-rotation

    new_tail=head
    for i in range(skip_len-1):
        new_tail=new_tail.next
    new_head=new_tail.next
    new_tail.next=None
    return new_head

# head=rotateByK(head,8)
# printLL(head)



