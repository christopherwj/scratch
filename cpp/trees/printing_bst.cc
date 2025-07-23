struct TreeNode {
    int item;
    TreeNode* left;
    TreeNode* right;
}

//let's recursively count the amount of nodes in the tree


int countNodes(TreeNode* root) {
    if(root == nullptr) 
        return 0;
    else{
        count = 1;
        count += countNodes(root->left);
        count += countNodes(root->right);
        return count;
    }
}

void preorderPrint(TreeNode* root) {
    if(root != nullptr) {
        cout << root->item << " ";
        preorderPrint(root->left);
        preorderPrint(root->right);
    }
}

