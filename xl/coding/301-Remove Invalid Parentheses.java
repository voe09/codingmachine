// backtracking, O(2 ^ N)
// 先算个能删多少，减少recursion
class Solution {
    private HashSet<String> validSet = new HashSet<String>();
    public List<String> removeInvalidParentheses(String s) {
        int len = s.length();
        int left = 0, right = 0;
        for (int i = 0; i < len; i++) {
            char c = s.charAt(i);
            if (c == '(') {
                left++;
            } else if (c == ')') {
                if (left > 0) {
                    left--;
                } else {
                    right++;
                }
            }
        }

        removeP(s, 0, 0, 0, left, right, new StringBuilder());
        return new ArrayList<String>(validSet);
    }

    private void removeP(String s, int idx, int curL, int curR, int toRemoveL, int toRemoveR, StringBuilder sb) {
        if (toRemoveL < 0 || toRemoveR < 0) {
            return;
        }
        if (idx == s.length()) {
            if (toRemoveL == 0 && toRemoveR == 0) {
            validSet.add(sb.toString());
            }
            return;
        }
        char c = s.charAt(idx);
        if (c == '(') {
            sb.append(c);
            removeP(s, idx + 1, curL + 1, curR, toRemoveL, toRemoveR, sb);
            sb.deleteCharAt(sb.length() - 1);
            removeP(s, idx + 1, curL, curR, toRemoveL - 1, toRemoveR, sb);
        } else if (c == ')') {
            if (curL > curR) {
                sb.append(c);
                removeP(s, idx + 1, curL, curR + 1, toRemoveL, toRemoveR, sb);
                sb.deleteCharAt(sb.length() - 1);
            }
            removeP(s, idx + 1, curL, curR, toRemoveL, toRemoveR - 1, sb);
        } else {
            sb.append(c);
            removeP(s, idx + 1, curL, curR, toRemoveL, toRemoveR, sb);
            sb.deleteCharAt(sb.length() - 1);
        }
    }
}