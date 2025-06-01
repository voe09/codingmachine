// O(N)
class Solution {
    public int countVowelSubstrings(String word) {
        // len > 0, only lowercase
        int res = 0;
        int len = word.length();
        HashSet<Character> set = new HashSet<Character>(Set.of('a', 'e', 'i', 'o', 'u'));
        HashMap<Character, Integer> count = new HashMap<>();
        int l = 0, r = 0, k = 0;
        while (r < len) {
            char c = word.charAt(r);
            if (set.contains(c)) {
                count.put(c, count.getOrDefault(c, 0) + 1);
                while (count.size() == 5) {
                    char rm = word.charAt(k);
                    if (count.get(rm) == 1) {
                        count.remove(rm);
                    } else {
                        count.put(rm, count.get(rm) - 1);
                    }
                    k++;
                }
                res += k - l;
            } else {
                l = r + 1;
                k = r + 1;
                count.clear();
            }
            r++;
        }
        
        return res;
    }
}