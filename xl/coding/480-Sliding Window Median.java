// O(N log K)，用TreeSet 
class Solution {
    int counter = 0;
    class DoubleWrapper {
        public double val;
        private int id;
        public DoubleWrapper(double val) {
            this.val = val;
            id = counter++;
        }
        public DoubleWrapper(int val) {
            this.val = val;
            id = counter++;
        }
    }
    public double[] medianSlidingWindow(int[] nums, int k) {
        TreeSet<DoubleWrapper> tSetL = new TreeSet<DoubleWrapper>((a, b) -> {
            if (a.val == b.val) {
                // TreeSet 用comparator判断是不是同一个元素，所以要加个ID
                return a.id - b.id;
            } else {
                return Double.compare(a.val, b.val);
            }
        });
        TreeSet<DoubleWrapper> tSetR = new TreeSet<DoubleWrapper>((a, b) -> {
            if (a.val == b.val) {
                return a.id - b.id;
            } else {
                return Double.compare(a.val, b.val);
            }
        });
        int len = nums.length;
        double[] res = new double[len - k + 1];
        DoubleWrapper[] records = new DoubleWrapper[len];
        for (int i = 0; i < k; i++) {
            records[i] = new DoubleWrapper(nums[i]);
            tSetL.add(records[i]);
        }
        for (int i = 0; i < k / 2; i++) {
            tSetR.add(tSetL.pollLast());
        }
        res[0] = k % 2 == 1 ? tSetL.last().val : (tSetL.last().val + tSetR.first().val) / 2.0;
        for (int i = k; i < len; i++) {
            if (records[i - k].val <= tSetL.last().val) {
                tSetL.remove(records[i - k]);
            } else {
                tSetR.remove(records[i - k]);
            }
            records[i] = new DoubleWrapper(nums[i]);
            tSetL.add(records[i]);
            tSetR.add(tSetL.pollLast());
            if (tSetL.size() < tSetR.size()) {
                tSetL.add(tSetR.pollFirst());
            }
            res[i - k + 1] = k % 2 == 1 ? tSetL.last().val : (tSetL.last().val + tSetR.first().val) / 2.0;
        }

        return res;
    }
}

//////////////////////////////////////////////
// O(N log N), 2 heaps
class Solution {
    public double[] medianSlidingWindow(int[] nums, int k) {
        PriorityQueue<Double> pqL = new PriorityQueue<Double>((a, b) -> {
            return Double.compare(b, a);
        });
        PriorityQueue<Double> pqR = new PriorityQueue<Double>();
        int len = nums.length;
        double[] res = new double[len - k + 1];
        if (k == 1) {
            for (int i = 0; i < len; i++) {
                res[i] = nums[i];
            }
            return res;
        }
        double isEven = (k + 1) % 2;
        double isOdd = k % 2;
        HashMap<Double, Integer> removed = new HashMap<Double, Integer>();
        for (int i = 0; i < k; i++) {
            pqL.offer((double)nums[i]);
        }
        for (int i = 0; i < k / 2; i++) {
            pqR.offer(pqL.poll());
        }
        res[0] = (double) (pqL.peek() + pqR.peek() * isEven + pqL.peek() * isOdd) / 2.0;
        for (int i = 1; i < res.length; i++) {
            double remove = nums[i - 1];
            double add = nums[i + k - 1];
            int balance = 0;
            if (remove > res[i - 1]) {
                balance++;
                // if (pqR.peek() == remove) {
                //     pqR.poll();
                // } else {
                removed.put(remove, removed.getOrDefault(remove, 0) + 1);
            } else {
                balance--;
                // if (pqL.peek() == remove) {
                //     pqL.poll();
                // } else {
                removed.put(remove, removed.getOrDefault(remove, 0) + 1);
            }
            if (add > res[i - 1]) {
                balance--;
                pqR.offer(add);
            } else {
                balance++;
                pqL.offer(add);
            }

            if (balance > 0) {
                pqR.offer(pqL.poll());
            } else if (balance < 0) {
                pqL.offer(pqR.poll());
            }
            // 先搞pqL，因为用的pqL多放
            while (removed.containsKey(pqL.peek()) && removed.get(pqL.peek()) > 0) {
                double removing = pqL.poll();
                removed.put(removing, removed.get(removing) - 1);
            }
            while (removed.containsKey(pqR.peek()) && removed.get(pqR.peek()) > 0) {
                double removing = pqR.poll();
                removed.put(removing, removed.get(removing) - 1);
            }
            res[i] = (double)pqL.peek() + (double)pqR.peek() * isEven + (double)pqL.peek() * isOdd;
            res[i] /= 2.0;
        }

        return res;
    }
}