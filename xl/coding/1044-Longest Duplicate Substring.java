class Solution {
    final int MOD = (int)Math.pow(10, 9) + 7;
    String res = "";
    public String longestDupSubstring(String s) {
        // len > 0
        int len = s.length();
        int[] nums = new int[len];
        for (int i = 0; i < len; i++) {
            char c = s.charAt(i);
            nums[i] = c - 'a';
        }

        int start = 1, end = len;
        int mid = 0;
        while (start <= end) {
            mid = (end - start) / 2 + start;
            if (hasDup(s, nums, mid)) {
                start = mid + 1;
            } else {
                end = mid - 1;
            }
        }
        hasDup(s, nums, start - 1);
        return res;
    }

    private boolean hasDup(String s, int[] nums, int tLen) {
        HashMap<Long, List<Integer>> found = new HashMap<>();
        long hash = nums[0], wei = 1;
        for (int i = 1; i < tLen; i++) {
            hash = (hash * 26 + nums[i]) % MOD;
            wei = (wei * 26) % MOD;
        }
        found.computeIfAbsent(hash, k -> new ArrayList<Integer>()).add(0);
        for (int i = tLen; i < nums.length; i++) {
            hash = (hash - nums[i - tLen] * wei % MOD + MOD) % MOD;
            hash = (hash * 26 + nums[i]) % MOD;
            if (found.containsKey(hash)) {
                List<Integer> startList = found.get(hash);
                for (int start : startList) {
                    if (s.substring(start, start + tLen).equals(s.substring(i - tLen + 1, i + 1))) {
                        res = s.substring(start, start + tLen);
                        return true;
                    }
                }
            }
            found.computeIfAbsent(hash, k -> new ArrayList<Integer>()).add(i - tLen + 1);
        }
        return false;
    }
}