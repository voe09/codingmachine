// O(N)
// or just compare two HashMap (s' and p's)
class Solution {
    public List<Integer> findAnagrams(String s, String p) {
        int lens = s.length(), lenp = p.length();
        HashMap<Character, Integer> count = new HashMap<>();
        int seen = 0;
        for (char c : p.toCharArray()) {
            count.put(c, count.getOrDefault(c, 0) + 1);
        }
        int l = 0, r = 0;
        List<Integer> res = new ArrayList<>();
        while (r < lens) {
            char c = s.charAt(r);
            if (count.containsKey(c)) {
                while (count.get(c) == 0) {
                    if (count.get(s.charAt(l)) == 0) {
                        seen--;
                    }
                    count.put(s.charAt(l), count.get(s.charAt(l)) + 1);
                    l++;
                }
                count.put(c, count.get(c) - 1);
                if (count.get(c) == 0) {
                    seen++;
                }
            } else {
                while (l < r) {
                    char start = s.charAt(l);
                    if (count.get(start) == 0) {
                        seen--;
                    }
                    count.put(start, count.get(start) + 1);
                    l++;
                }
                l = r + 1;
            }
            if (r - l + 1 == lenp && seen == count.size()) {
                res.add(l);
                count.put(s.charAt(l), count.get(s.charAt(l)) + 1);
                seen--;
                l++;
            }
            r++;
        }
        return res;
    }
}