class Node :
    def __init__(self,data):
        self.data=data
        self.next=None


class linked_list:

    def __init__(self):
        self.head = None

    def append(self, List):

        if len(List) <= 0:
            return

        new_node = Node(List[0])
        if self.head is None:
            self.head = new_node
        else:
            new_node.next = self.head
            self.head = new_node

        self.append(List[1:])

    def Print(self,curr):
        print(curr.data)
        while curr.next is None :
            return
        curr=curr.next
        self.Print(curr)

    def reverse(self):
        prev=None
        curr=self.head

        while curr is not None :
            next=curr.next
            curr.next=prev
            prev=curr
            curr=next
        self.head=prev

    def detect_loop(self):
        slow_p=self.head
        fast_p=self.head

        while slow_p and fast_p and fast_p.next :
            slow_p=slow_p.next
            fast_p=fast_p.next.next

            if slow_p==fast_p :
                return self.remove_loop(slow_p)
        return False

    def remove_loop(self,loop_node):
        ptr_1=self.head
        ptr_2=loop_node

        while(1):
            if ptr_1.next==ptr_2.next :
                print(ptr_1.next.data)
                ptr_2.next=None
                return True
            ptr_1=ptr_1.next
            ptr_2=ptr_2.next

    def find_middle(self):
        slow_p=self.head
        fast_p=self.head
        while (fast_p.next != None and
               fast_p.next.next != None):
            slow_p=slow_p.next
            fast_p=fast_p.next.next
        print(slow_p.data)

    def remove_duplicate(self):
        curr=second=self.head
        while curr is not None :
            while second.next is not None :
                if curr.data == second.next.data :
                    second.next=second.next.next
                else:
                    second=second.next
            curr=curr.next
            second=curr

    def middle(self,node):
        slow_p=fast_p=node

        while fast_p.next and fast_p.next.next :
            slow_p=slow_p.next
            fast_p=fast_p.next.next
        return slow_p

    def sorted_merge(self,left,right):
        result=None
        if left is None :
            return right
        if right is None :
            return left
        if left.data <= right.data :
            result=left
            result.next=self.sorted_merge(left.next,right)
        else :
            result=right
            result.next=self.sorted_merge(left,right.next)
        return result

    def merge_sort(self,node):
        if node.next is None :
            return node

        mid=self.middle(node)
        mid_next=mid.next
        mid.next=None
        left=self.merge_sort(node)
        right=self.merge_sort(mid_next)
        return self.sorted_merge(left,right)

    def intersection(self,a,b):
        result=None
        if a is None or b  is None :
            return None

        if  a.data==b.data :
            result=a
            result.next=self.intersection(a.next,b.next)
        elif a.data > b.data :
            result=self.intersection(a,b.next)
        else:
            result=self.intersection(a.next,b)
        return result

    def union(self,a,b):
        result=None

        if a is None :
            return b
        if b is None :
            return a
        if a.data==b.data :
            result=a
            result.next=self.union(a.next,b.next)
        elif a.data > b.data :
            result=b
            result.next=self.union(a,b.next)
        else:
            result=a
            result.next=self.union(a.next,b)
        return result

    def n_from_end(self,n):
        main_p=ref_p=self.head

        while main_p and n >0 :
            main_p=main_p.next
            n=n-1
        if n<= 0 :
            while main_p :
                main_p=main_p.next
                ref_p=ref_p.next
            return ref_p.data
        else :
            print("N is greater than length")
            return

def length(node):
        len=0
        while node :
            len+=1
            node=node.next
        return len

def find_intersection_by_diffrence(a,b):
    l1=length(a)
    l2=length(b)

    if l1 > l2 :
        diff=l1-l2
        large=a
        small=b
    else :
        diff=l2-l1
        large=b
        small=a
    while diff > 0 :
        large=large.next
        diff-=1

    while large is not small :
        large=large.next
        small=small.next

    if large.data == small.data :
        return small.data
    else :
        print("No Intersection")

def find_intersection_by_circle_loop(a,b):
    l1=1
    point_a=a
    while point_a.next :
        l1+=1
        point_a=point_a.next
    point_a.next=a
    first=b
    second=b
    while l1>0 :
        first=first.next
        l1-=1
    while first!=second :
        first=first.next
        second=second.next
    return first.data

def delete_node_greater_value_on_right_side(llist) :
    llist.reverse()
    curr=llist.head
    max=llist.head.data
    while curr.next :
        if curr.next.data < max :
            curr.next=curr.next.next
        else :
            max=curr.next.data
            curr=curr.next
    llist.reverse()

def segregate_even_odd(node):

    prev=None
    curr=node
    end=node
    while end.next :
        end=end.next
    new_end=end
    while curr.data%2!=0 and curr!=end :
        new_end.next=curr
        curr=curr.next
        new_end.next.next=None
        new_end=new_end.next

    if  curr.data%2==0 :
        new_head=curr

        while curr!=end :

            if curr.data%2==0 :
                prev=curr
                curr=curr.next
            else:
                prev.next=curr.next
                curr.next=None
                new_end.next=curr
                new_end=curr
                curr=prev.next
    else:
        prev=curr
        new_head=prev.next
    if curr==end and curr.data%2!=0 :
        prev.next=end.next
        new_end.next=curr
        curr.next=None
    return new_head

def find_triple(a,b,c,num):
    while a :
        while b and c :
            sum=a.data+b.data+c.data
            if sum==num:
                print(a.data,b.data,c.data)
                return
            elif sum > num :
                b=b.next
            else:
                c=c.next
        a=a.next

    print("No Triplet found")

def reverse(head):
    prev=None
    curr=head

    while curr is not None :
        next=curr.next
        curr.next=prev
        prev=curr
        curr=next
    return prev

def find_middle(head):
    slow=fast=head
    while fast.next and fast.next.next:
        slow=slow.next
        fast=fast.next.next
    return slow

def rearrange(head):
    middle_p=find_middle(head)
    temp=middle_p.next
    middle_p.next=None
    left_p=head
    right_p=reverse(temp)
    while left_p and right_p :
        temp=right_p
        right_p=right_p.next
        temp.next=left_p.next
        left_p.next=temp
        left_p=temp.next

def zigZaglist(head):
    curr=prev=head
    flag=1
    while curr.next:

        if (flag is 1 and curr.data > curr.next.data) or (flag is 2 and curr.data<curr.next.data) :
            # print(curr.data)
            temp=curr.next
            # print(temp.data)
            curr.next=temp.next
            temp.next=curr
            if prev!=curr:
                prev.next=temp
            else:
                head=temp
            curr=temp
        prev=curr
        curr=curr.next
        if flag is 1 :
            flag=2
        else:
            flag=1
    return head

def sorted_merge(a,b):
    if  a is None :
        return b
    if b is None :
        return a
    if a.data>b.data:
        head=Node(b.data)
        head.next=sorted_merge(a,b.next)
    else:
        head=Node(a.data)
        head.next=sorted_merge(a.next,b)
    return head

def merge_k_lists(arr,n):
    if n is 1:
        return arr[0]
    if n is 0:
        return None
    head1=sorted_merge(arr[0],arr[1])
    return sorted_merge(head1,merge_k_lists(arr[2:],n-2))

def rotate_helper(head,tail,shift):
    tail.next=head
    while shift>0 :
        head=head.next
        shift-=1
    return head

def rotate_Blockwise(head,l,d):
    count=1
    curr=head
    if head is None or head.next is None:
        return head

    while count<l and curr.next :
        curr=curr.next
        count+=1
    next_Node=curr.next
    if d>0:
        temp=rotate_helper(head,curr,count-(d%l)-1)
    else:
        temp=rotate_helper(head,curr,((-d)%l)-1)
    head=temp.next
    temp.next=rotate_Blockwise(next_Node,l,d)
    return head


llist = linked_list()
a=[10,4,5,6,7,8,9]
llist.append(a)
llist.find_middle()
