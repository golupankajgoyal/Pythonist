def merge(l1, l2):
    if not l1 or not l2:
        return l1 or l2

    if l1.val < l2.val:
        l1.next = merge(l1.next, l2)
        return l1
    else:
        l2.next = merge(l1, l2.next)
        return l2


def findZeroSubarray(head):
    arr = []
    node_list = []
    temp = head
    while temp:
        arr.append(temp.val)
        node_list.append(temp)
        temp = temp.next
    n= len(arr)
    for i in range(n):
        if arr[i] == 0:
            continue
        sum = 0
        for j in range(i, n):
            sum += arr[j]
            if sum == 0:
                for k in range(i, j + 1):
                    arr[k] = 0
                break
    ans = curr = ListNode()
    for i in range(n):
        if arr[i] != 0:
            curr.next = node_list[i]
            curr = curr.next
    curr.next = None
    return ans.next



arr= [1,2,3,-3,-2,0,1,-1]
findZeroSubarray(arr, len(arr))
