import streamlit as st
import random
import pandas as pd
from collections import Counter, defaultdict
from typing import List, Dict, Optional
from itertools import chain
import logging
import os

# ==============================================================================
# 1. إعدادات النظام
# ==============================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("JordanLottery")

class LotteryConfig:
    MIN_NUM = 1
    MAX_NUM = 32
    DEFAULT_TICKET_SIZE = 6
    MIN_TICKET_SIZE = 6
    MAX_TICKET_SIZE = 10
    MAX_GENERATION_ATTEMPTS = 50000 
    STRICT_SHADOW_ATTEMPTS = 15000
    DEFAULT_SUM_TOLERANCE = 0.15
    MAX_BATCH_SIZE = 10

class MockDataGenerator:
    @staticmethod
    def create_mock_history(rows: int = 248) -> pd.DataFrame:
        data = []
        for i in range(1, rows + 1):
            nums = sorted(random.sample(range(LotteryConfig.MIN_NUM, LotteryConfig.MAX_NUM + 1), LotteryConfig.DEFAULT_TICKET_SIZE))
            data.append({'draw_id': i, 'numbers': nums})
        return pd.DataFrame(data)

# ==============================================================================
# 2. المحلل الإحصائي (Analyzer)
# ==============================================================================
class LotteryAnalyzer:
    def __init__(self, history_df: pd.DataFrame):
        self.history_df = history_df
        self.past_draws_sets = [set(nums) for nums in history_df['numbers']]
        self.draw_map = {row['draw_id']: row['numbers'] for _, row in history_df.iterrows()}
        
        self.number_to_draws_index = defaultdict(set)
        for idx, draw_set in enumerate(self.past_draws_sets):
            for num in draw_set:
                self.number_to_draws_index[num].add(idx)
        
        all_numbers = list(chain.from_iterable(history_df['numbers']))
        self.frequency = Counter(all_numbers)
        self.total_draws = len(history_df)
        
        all_sums = [sum(nums) for nums in history_df['numbers']]
        self.global_avg_sum = sum(all_sums) / len(all_sums) if all_sums else 0
        
        sorted_nums = sorted(range(LotteryConfig.MIN_NUM, LotteryConfig.MAX_NUM + 1), 
                           key=lambda x: self.frequency[x], reverse=True)
        self.hot_pool = set(sorted_nums[:16])
        self.cold_pool = set(sorted_nums[16:])

    def calculate_custom_average(self, mode: str, param1: int = 0, param2: int = 0) -> tuple:
        """يعيد (المتوسط, قائمة المجاميع للرسم البياني)"""
        df = self.history_df.copy()
        if mode == "Last N Draws":
            if param1 > len(df): param1 = len(df)
            df = df.iloc[-param1:]
        elif mode == "Specific Range":
            df = df[(df['draw_id'] >= param1) & (df['draw_id'] <= param2)]
        
        if df.empty: return self.global_avg_sum, []
        
        sums = [sum(nums) for nums in df['numbers']]
        avg = sum(sums) / len(sums) if sums else 0
        return avg, sums

    def get_ticket_profile(self, ticket: List[int]) -> str:
        hot_count = sum(1 for n in ticket if n in self.hot_pool)
        total = len(ticket)
        if hot_count >= total * 0.7: return "🔥 ساخنة"
        elif hot_count <= total * 0.3: return "❄️ باردة"
        else: return "⚖️ متوازنة"
    
    def get_numbers_from_draw(self, draw_id: int) -> Optional[List[int]]:
        return self.draw_map.get(draw_id)

# ==============================================================================
# 3. المدقق (Validator)
# ==============================================================================
class TicketValidator:
    @staticmethod
    def count_sequences(numbers: List[int]) -> int:
        sorted_nums = sorted(numbers)
        return sum(1 for i in range(len(sorted_nums) - 1) if sorted_nums[i + 1] == sorted_nums[i] + 1)

    @staticmethod
    def count_shadow_occurrences(numbers: List[int]) -> int:
        units = [n % 10 for n in numbers]
        return sum(count - 1 for count in Counter(units).values() if count >= 2)

    @staticmethod
    def check_anti_match_optimized(ticket_set: set, analyzer: 'LotteryAnalyzer', limit: int) -> bool:
        candidate_indices = set()
        for num in ticket_set:
            candidate_indices.update(analyzer.number_to_draws_index.get(num, set()))
        for idx in candidate_indices:
            if len(ticket_set & analyzer.past_draws_sets[idx]) >= limit:
                return False
        return True

    @staticmethod
    def analyze_ticket(numbers: List[int], analyzer: Optional['LotteryAnalyzer'] = None) -> Dict:
        analysis = {
            'sum': sum(numbers),
            'sequences': TicketValidator.count_sequences(numbers),
            'shadows': TicketValidator.count_shadow_occurrences(numbers),
            'odd': sum(1 for n in numbers if n % 2 != 0),
        }
        if analyzer:
            analysis['profile'] = analyzer.get_ticket_profile(numbers)
        return analysis

# ==============================================================================
# 4. محرك التوليد (The Generator)
# ==============================================================================
class TicketGenerator:
    def __init__(self, analyzer: LotteryAnalyzer):
        self.analyzer = analyzer
        self.full_pool = list(range(LotteryConfig.MIN_NUM, LotteryConfig.MAX_NUM + 1))

    def _validate_criteria(self, criteria: Dict) -> List[str]:
        errors = []
        size = criteria.get('size', 6)
        inc_draw = criteria.get('include_from_draw')
        inc_cnt = criteria.get('include_count', 0)
        if inc_draw is not None and inc_cnt > 0:
            if inc_cnt > size: errors.append(f"❌ لا يمكن اقتباس {inc_cnt} أرقام في تذكرة حجمها {size}")
            if not self.analyzer.get_numbers_from_draw(inc_draw): errors.append(f"❌ السحب {inc_draw} غير موجود")
        if not (LotteryConfig.MIN_TICKET_SIZE <= size <= LotteryConfig.MAX_TICKET_SIZE):
            errors.append(f"❌ الحجم {size} غير صالح")
        return errors

    def estimate_success_probability(self, criteria: Dict, sample_size: int = 2000) -> Dict:
        size = criteria.get('size', 6)
        passed = 0
        target_avg = criteria.get('target_average', self.analyzer.global_avg_sum)
        sum_tolerance = LotteryConfig.DEFAULT_SUM_TOLERANCE

        for _ in range(sample_size):
            candidate = sorted(random.sample(self.full_pool, size))
            
            inc_draw = criteria.get('include_from_draw')
            inc_cnt = criteria.get('include_count', 0)
            if inc_draw and inc_cnt > 0:
                draw_nums = self.analyzer.get_numbers_from_draw(inc_draw)
                if draw_nums:
                    intersect = len(set(candidate) & set(draw_nums))
                    if intersect != inc_cnt: continue 

            if criteria.get('sequences_count') is not None and TicketValidator.count_sequences(candidate) != criteria['sequences_count']: continue
            if criteria.get('odd_count') is not None and sum(1 for n in candidate if n%2!=0) != criteria['odd_count']: continue
            
            if criteria.get('shadows_count') is not None:
                curr = TicketValidator.count_shadow_occurrences(candidate)
                if not (max(0, criteria['shadows_count']-1) <= curr <= criteria['shadows_count']+1): continue
            
            if criteria.get('sum_near_avg'):
                s = sum(candidate)
                if not (target_avg * (1-sum_tolerance) <= s <= target_avg * (1+sum_tolerance)): continue

            passed += 1
        
        prob = (passed / sample_size) * 100
        advice = "✅ سهلة" if prob > 5 else ("⚡ متوسطة" if prob > 0.5 else "⚠️ صعبة جداً")
        return {"probability": round(prob, 2), "advice": advice}

    def _generate_single_ticket(self, criteria: Dict, sum_tolerance: float) -> Dict:
        try:
            size = criteria.get('size', 6)
            strategy = criteria.get('strategy', 'Any')
            target_avg = criteria.get('target_average', self.analyzer.global_avg_sum)
            
            include_draw_id = criteria.get('include_from_draw')
            include_count = criteria.get('include_count', 0)
            forced_numbers = []
            forbidden_numbers = set()

            if include_draw_id and include_count > 0:
                source_nums = self.analyzer.get_numbers_from_draw(include_draw_id)
                if source_nums:
                    forced_numbers = random.sample(source_nums, min(len(source_nums), include_count))
                    forbidden_numbers = set(source_nums) - set(forced_numbers)
            
            pool_source = self.full_pool
            if strategy == 'Hot': pool_source = list(self.analyzer.hot_pool)
            elif strategy == 'Cold': pool_source = list(self.analyzer.cold_pool)
            
            current_pool = [n for n in pool_source if n not in forced_numbers and n not in forbidden_numbers]
            needed_count = size - len(forced_numbers)
            
            if len(current_pool) < needed_count:
                current_pool = [n for n in self.full_pool if n not in forced_numbers and n not in forbidden_numbers]

            for attempt in range(LotteryConfig.MAX_GENERATION_ATTEMPTS):
                random_part = random.sample(current_pool, needed_count)
                candidate = sorted(forced_numbers + random_part)
                
                if strategy == 'Balanced':
                    hot_in_ticket = sum(1 for n in candidate if n in self.analyzer.hot_pool)
                    half = size / 2
                    if not (half - 1 <= hot_in_ticket <= half + 1): continue

                if criteria.get('sequences_count') is not None and TicketValidator.count_sequences(candidate) != criteria['sequences_count']: continue
                if criteria.get('odd_count') is not None and sum(1 for n in candidate if n%2!=0) != criteria['odd_count']: continue
                
                if criteria.get('shadows_count') is not None:
                    curr = TicketValidator.count_shadow_occurrences(candidate)
                    if attempt < LotteryConfig.STRICT_SHADOW_ATTEMPTS:
                        if curr != criteria['shadows_count']: continue
                    else:
                        if not (max(0, criteria['shadows_count']-1) <= curr <= criteria['shadows_count']+1): continue
                
                if criteria.get('sum_near_avg'):
                    s = sum(candidate)
                    if not (target_avg * (1-sum_tolerance) <= s <= target_avg * (1+sum_tolerance)): continue
                
                if criteria.get('anti_match_limit') and not TicketValidator.check_anti_match_optimized(set(candidate), self.analyzer, criteria['anti_match_limit']): continue
                
                return {"status": "success", "ticket": candidate, "attempts": attempt + 1}
            
            return {"status": "error", "reason": "استنفاد المحاولات"}
        except Exception as e:
            return {"status": "error", "reason": f"خطأ داخلي: {str(e)}"}

    def generate_batch(self, criteria: Dict, count: int = 1) -> Dict:
        errors = self._validate_criteria(criteria)
        if errors: return {"status": "validation_error", "errors": errors}
        
        actual_count = min(count, LotteryConfig.MAX_BATCH_SIZE)
        generated_tickets = []
        seen_signatures = set()
        errors_list = []
        
        for i in range(actual_count):
            for retry in range(50):
                res = self._generate_single_ticket(criteria, LotteryConfig.DEFAULT_SUM_TOLERANCE)
                if res['status'] == 'success':
                    sig = tuple(res['ticket'])
                    if sig not in seen_signatures:
                        seen_signatures.add(sig)
                        anl = TicketValidator.analyze_ticket(res['ticket'], self.analyzer)
                        generated_tickets.append({"id": i+1, "numbers": res['ticket'], "analysis": anl, "attempts": res['attempts']})
                        break
                else:
                    if retry == 0: errors_list.append(res['reason'])
        
        status = "success" if len(generated_tickets) == actual_count else ("partial_success" if generated_tickets else "failed")
        return {"status": status, "requested": count, "generated": len(generated_tickets), "tickets": generated_tickets, "errors": Counter(errors_list).most_common(3)}

# ==============================================================================
# 5. واجهة المستخدم (v5.2 - Charts & Precision)
# ==============================================================================
def load_data(uploaded_file=None):
    df = None
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        except Exception as e: return None, f"خطأ: {e}"
    else:
        for fname in ['data.xlsx', 'data.csv', 'lotto.xlsx', 'lotto.csv']:
            if os.path.exists(fname):
                try: df = pd.read_csv(fname) if fname.endswith('.csv') else pd.read_excel(fname); break 
                except: continue
        if df is None: return None, "No local file"

    if df is not None:
        try:
            # Smart Column Detection
            if not {'N1', 'N2', 'N3', 'N4', 'N5', 'N6'}.issubset(df.columns):
                return None, "الملف يجب أن يحتوي على N1..N6"
            
            df['numbers'] = df[['N1','N2','N3','N4','N5','N6']].values.tolist()
            df['numbers'] = df['numbers'].apply(lambda x: sorted([int(n) for n in x]))
            
            # Handle Arabic or English ID
            if 'رقم السحب' in df.columns:
                df = df.rename(columns={'رقم السحب': 'draw_id'})
            elif 'DrawID' in df.columns: 
                df = df.rename(columns={'DrawID': 'draw_id'})
            elif 'draw_id' not in df.columns:
                df['draw_id'] = range(1, len(df)+1)
                
            return df, "Success"
        except Exception as e: return None, f"خطأ البيانات: {e}"
            
    return None, "No data"

def main():
    st.set_page_config(page_title="نظام لوتري الأردن الذكي", page_icon="🎰", layout="wide", initial_sidebar_state="expanded")
    st.markdown(\"\"\"<style>.main {direction: rtl;} h1,h2,h3,p,div,label,span {text-align: right; font-family: 'Segoe UI', sans-serif;} .stMetric {text-align: right !important;} .footer {position: fixed; left: 0; bottom: 0; width: 100%; background-color: #f0f2f6; color: #333; text-align: center; padding: 10px; border-top: 1px solid #ddd; font-size: 14px; z-index: 999; font-family: 'Segoe UI', sans-serif; font-weight: bold;} @media (prefers-color-scheme: dark) { .footer {background-color: #0e1117; color: #888; border-top: 1px solid #333;} }</style>\"\"\", unsafe_allow_html=True)

    st.title("🎰 نظام لوتري الأردن الذكي (v5.2 Visuals)")
    
    with st.sidebar:
        st.header("1. إعدادات البيانات")
        uploaded_file = st.file_uploader("تحديث البيانات", type=['xlsx', 'csv'])
        df, msg = load_data(uploaded_file)
        
        if df is not None:
            if uploaded_file: st.success("✅ ملف جديد")
            else: st.info("📂 ملف تلقائي")
            
            st.session_state.history_df = df
            data_hash = hash(str(df.values.tobytes()))
            if 'data_hash' not in st.session_state or st.session_state.data_hash != data_hash:
                st.session_state.analyzer = LotteryAnalyzer(df)
                st.session_state.generator = TicketGenerator(st.session_state.analyzer)
                st.session_state.data_hash = data_hash
            
            analyzer = st.session_state.analyzer
            generator = st.session_state.generator
            st.metric("إجمالي السحوبات", analyzer.total_draws)
            st.metric("المتوسط العام", f"{analyzer.global_avg_sum:.2f}")
            
        else:
            st.warning("⚠️ بيانات وهمية (للتجربة)")
            st.session_state.history_df = MockDataGenerator.create_mock_history(248)
            st.session_state.analyzer = LotteryAnalyzer(st.session_state.history_df)
            st.session_state.generator = TicketGenerator(st.session_state.analyzer)
            analyzer = st.session_state.analyzer
            generator = st.session_state.generator

    if 'history_df' in st.session_state:
        col1, col2 = st.columns([1, 1.5])
        with col1:
            st.subheader("⚙️ إعدادات التوليد")
            strategy = st.selectbox("🎯 استراتيجية الأرقام", ["Any (الكل)", "Hot (ساخنة)", "Cold (باردة)", "Balanced (متوازنة)"])
            strategy_map = {"Any (الكل)": "Any", "Hot (ساخنة)": "Hot", "Cold (باردة)": "Cold", "Balanced (متوازنة)": "Balanced"}
            
            with st.container(border=True):
                st.markdown("**📊 ضبط المتوسط الحسابي**")
                avg_chk = st.checkbox("الالتزام بالمتوسط", value=True)
                target_avg_val = analyzer.global_avg_sum
                chart_data = [] # للرسم البياني
                
                if avg_chk:
                    avg_mode = st.selectbox("المرجع لحساب المتوسط:", ["كافة السحوبات (Default)", "آخر N سحب", "نطاق محدد"])
                    if avg_mode == "آخر N سحب":
                        n_draws = st.number_input("عدد السحوبات الأخيرة", 5, analyzer.total_draws, 20)
                        target_avg_val, chart_data = analyzer.calculate_custom_average("Last N Draws", param1=n_draws)
                        st.caption(f"متوسط آخر {n_draws} سحب: **{target_avg_val:.2f}**")
                    elif avg_mode == "نطاق محدد":
                        c1, c2 = st.columns(2)
                        start_d = c1.number_input("من سحب", 1, analyzer.total_draws, max(1, analyzer.total_draws-50))
                        end_d = c2.number_input("إلى سحب", 1, analyzer.total_draws, analyzer.total_draws)
                        target_avg_val, chart_data = analyzer.calculate_custom_average("Specific Range", param1=start_d, param2=end_d)
                        st.caption(f"المتوسط للنطاق: **{target_avg_val:.2f}**")
                    else:
                        target_avg_val, chart_data = analyzer.calculate_custom_average("All")
                        st.caption(f"المتوسط العام: **{target_avg_val:.2f}**")
                    
                    # عرض الرسم البياني للتأكيد
                    if chart_data:
                        st.line_chart(chart_data, height=150)
                        st.caption("📈 تذبذب مجموع الأرقام في النطاق المختار")

            with st.container(border=True):
                t_count = st.number_input("عدد التذاكر", 1, 10, 3)
                t_size = st.slider("حجم التذكرة", 6, 10, 6)
                odd = st.number_input("عدد الفردي", 0, t_size, t_size//2)
                seq = st.number_input("عدد المتتاليات", 0, t_size-1, 0)
                sha = st.number_input("عدد الظلال", 0, 3, 1)

            with st.container(border=True):
                st.markdown("**🔄 تكرار صارم (Pivot)**")
                use_past = st.checkbox("تثبيت أرقام من سحب سابق")
                inc_draw = None; inc_cnt = 0
                if use_past:
                    c1, c2 = st.columns(2)
                    inc_draw = c1.number_input("رقم السحب", 1, analyzer.total_draws, analyzer.total_draws)
                    inc_cnt = c2.number_input("عدد الأرقام", 1, min(6, t_size), 1)
                    past_nums = analyzer.get_numbers_from_draw(inc_draw)
                    if past_nums: st.caption(f"أرقام السحب {inc_draw}: {past_nums}")

                st.markdown("---")
                anti = st.slider("تجنب تكرار سحوبات (X)", 3, t_size, 5)

            criteria = {
                'size': t_size, 'sequences_count': seq, 'odd_count': odd, 
                'shadows_count': sha, 'anti_match_limit': anti, 'sum_near_avg': avg_chk,
                'target_average': target_avg_val,
                'include_from_draw': inc_draw if use_past else None, 'include_count': inc_cnt if use_past else 0,
                'strategy': strategy_map[strategy]
            }

            if st.button("🔍 فحص الجدوى"):
                with st.spinner("جاري المحاكاة..."):
                    est = generator.estimate_success_probability(criteria)
                    color = "green" if est['probability'] > 5 else "red"
                    st.markdown(f"**النسبة:** :{color}[{est['probability']}%] ({est['advice']})")

            if st.button("🚀 توليد الآن", type="primary", use_container_width=True):
                with st.spinner("جاري التوليد..."):
                    res = generator.generate_batch(criteria, t_count)
                    st.session_state['last_result'] = res

        with col2:
            if 'last_result' in st.session_state:
                res = st.session_state['last_result']
                if res['status'] == 'validation_error':
                    st.error("خطأ:"); [st.write(f"- {e}") for e in res['errors']]
                elif res['status'] == 'failed':
                    st.error("فشل التوليد."); st.write("الأسباب:", res['errors'])
                else:
                    if res['status'] == 'partial_success': st.warning(f"تم توليد {res['generated']} تذاكر فقط.")
                    else: st.success(f"تم توليد {res['generated']} تذاكر بنجاح!")
                    
                    for t in res['tickets']:
                        with st.expander(f"🎫 تذكرة #{t['id']} - {t['analysis']['profile']}", expanded=True):
                            st.markdown("".join([f"<span style='display:inline-block; background:#dcfce7; color:#166534; padding:5px 10px; margin:2px; border-radius:50%; font-weight:bold; border:1px solid #166534'>{n}</span>" for n in t['numbers']]), unsafe_allow_html=True)
                            ca, cb, cc = st.columns(3)
                            ca.caption(f"المجموع: {t['analysis']['sum']}")
                            cb.caption(f"المتتاليات: {t['analysis']['sequences']}")
                            cc.caption(f"الظلال: {t['analysis']['shadows']}")
                            if use_past and inc_draw:
                                draw_nums = set(analyzer.get_numbers_from_draw(inc_draw))
                                matches = set(t['numbers']) & draw_nums
                                color = "green" if len(matches)==inc_cnt else "red"
                                st.markdown(f":{color}[✅ المطلوب: {inc_cnt} | 🎯 المحقق: {len(matches)} ({list(matches)})]")

    st.markdown(\"\"\"<div class="footer">برمجة وتطوير: <b>محمد العمري</b></div>\"\"\", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
