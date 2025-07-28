// can be modified to one pass
class Solution {
    public boolean checkValidString(String s) {
        int star = 0, count = 0;
        int len = s.length();
        // left -> right
        for (int i = 0; i < len; i++) {
            char c = s.charAt(i);
            if (c == '(') {
                count++;
            } else if (c == ')') {
                count--;
                if (count < 0) {
                    count = 0;
                    star--;
                    if (star < 0) {
                        return false;
                    }
                }
            } else {
                star++;
            }
        }
        count = 0;
        star = 0;
        // right -> left
        for (int i = len - 1; i >= 0; i--) {
            char c = s.charAt(i);
            if (c == ')') {
                count++;
            } else if (c == '(') {
                count--;
                if (count < 0) {
                    count = 0;
                    star--;
                    if (star < 0) {
                        return false;
                    }
                }
            } else {
                star++;
            }
        }
        return true;
    }
}