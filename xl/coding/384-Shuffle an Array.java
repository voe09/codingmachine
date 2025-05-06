// O(N)
class Solution {
    int[] original;
    Random rand;

    public Solution(int[] nums) {
        original = nums.clone();
        rand = new Random();
    }
    
    public int[] reset() {
        return original.clone();
    }

    public int[] shuffle() {
        int[] cur = original.clone();
        for (int i = 0; i < cur.length; i++) {
            int toSwap = rand.nextInt(cur.length - i) + i;
            swap(cur, i, toSwap);
        }

        return cur;
    }

    private void swap(int[] arr, int a, int b) {
        int tmp = arr[a];
        arr[a] = arr[b];
        arr[b] = tmp;
    }
}

/**
 * Your Solution object will be instantiated and called as such:
 * Solution obj = new Solution(nums);
 * int[] param_1 = obj.reset();
 * int[] param_2 = obj.shuffle();
 */