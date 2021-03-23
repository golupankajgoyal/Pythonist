from collections import defaultdict

def find_min_cameras(node,parent_cam,node_cam,dp):
    if  node==None:
        return inf
    if node.left==None and node.right==None:
        dp[node][1][1]=1
        dp[node][1][0]=0
        dp[node][0][1]=1
        dp[node][0][0]=inf
        return dp[node][parent_cam][node_cam]

    if dp[node][parent_cam][node_cam]!=-1:
        return dp[node][parent_cam][node_cam]

    if node_cam==1:
        result=min(find_min_cameras(node.left,1,1,dp),find_min_cameras(node.left,1,0,dp))\
               +min(find_min_cameras(node.right,1,1,dp),find_min_cameras(node.right,1,0,dp))+1
    else:
        if parent_cam==1:
            result = min(find_min_cameras(node.left, 0, 1, dp), find_min_cameras(node.left, 0, 0, dp)) \
                     + min(find_min_cameras(node.right,0, 1, dp), find_min_cameras(node.right,0, 0, dp))
        else:
            result = min(find_min_cameras(node.right,0,1,dp)+min(find_min_cameras(node.left, 0, 1, dp),
                                                                 find_min_cameras(node.left, 0, 0, dp)),
                         find_min_cameras(node.left,0,1,dp) + min(find_min_cameras(node.right, 0, 1, dp),
                                                                find_min_cameras(node.right, 0, 0, dp)))
    dp[node][parent_cam][node_cam]=result
    return result

dp=defaultdict(lambda :[[-1 for i in range(2)]for j in range(2)])
result=min(find_min_cameras(root,0,0,dp),find_min_cameras(root,0,1,dp))


