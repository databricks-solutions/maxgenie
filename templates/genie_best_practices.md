# Genie Space Optimization Best Practices

## Priority Framework (Most to Least Effective)

Apply changes in this order. Higher-priority levers have more impact on accuracy.

| Priority | Lever | Impact | MaxGenie Coverage |
|----------|-------|--------|-------------------|
| 1 | Data model quality (table/column descriptions, naming, hidden columns) | Highest | `curate --mode descriptions`, `curate --mode hide_noise` |
| 2 | Entity matching + format assistance on categorical/date columns | High | `curate --mode entity_matching`, `curate --mode format_assistance` |
| 3 | Join specifications between tables | High | Manual edit in `space/instructions/join_specs/` |
| 4 | SQL expressions (measures, filters, dimensions) | Medium-High | Manual edit in `space/instructions/sql_snippets/` |
| 5 | SQL examples (trusted example queries) | Medium | Manual edit in `space/instructions/examples/` |
| 6 | Text instructions (general guidance) | Lowest | Manual edit in `space/instructions/text_instruction.md` |

**One high-quality SQL example teaches Genie more effectively than 10 lines of text instructions.**

---

## 1. Data Model Quality (Priority 1)

### Table Count
- **5-7 tables optimal** for accuracy
- 8-15 acceptable for complex domains
- 16-25: hallucination risk increases significantly
- **30 tables hard maximum**
- If you need more, split into multiple focused spaces

### Column Management
- **25 or fewer columns per table** ideally
- **Hide irrelevant columns**: row IDs, hashes, ETL timestamps, internal flags
- Use descriptive names, not abbreviations (`customer_total_revenue` not `cust_tot_rev`)

### Table Descriptions
- Describe what the table represents in the context of *this space*
- Include which concepts/entities the table captures
- Reference the business domain the table serves

### Column Descriptions
- Provide context a user wouldn't infer from the column name
- For ambiguous columns: explain exactly what values mean
- For filter columns: enumerate valid values (e.g., "Values: Active, Inactive, Suspended")
- For date columns: explain the business meaning ("Transaction settlement date, not order date")
- For metric columns: define the calculation or business rule
- **Do not use vague auto-generated descriptions** — they add noise, not signal

### Synonyms
- Map business terms to column names (e.g., "revenue" -> `total_sales_amount`)
- Include common abbreviations used by the target audience
- Add industry-specific jargon mappings

---

## 2. Entity Matching & Format Assistance (Priority 2)

### Entity Matching
- Enable on **all categorical columns** where users reference specific values
- Best candidates: state/country codes, product categories, status codes, department names, customer segments
- Maps user terms to actual data values (e.g., "Florida" -> "FL", "Yes" -> "Y")
- **Limits**: 120 columns per space, 1,024 distinct values per column, 127 chars per value
- **Refresh when underlying data changes** — stale value lists cause mismatches
- String columns only; tables with row filters/column masks are excluded

### Format Assistance
- Provides representative values for eligible columns
- Helps Genie understand data types, patterns, and formatting
- Enable on date/time, currency, and coded columns
- **Disabling format assistance automatically disables entity matching** on that column

---

## 3. Join Specifications (Priority 3)

### When to Define Joins
- Always explicitly define relationships between tables in the space
- Don't rely on Genie inferring joins from column names (it may guess wrong)
- Unity Catalog PK/FK relationships are automatically imported

### Join Definition Structure
```yaml
left_table: orders
right_table: customers
join_condition: orders.customer_id = customers.customer_id
relationship: many_to_one
```

### Cardinality Types
- **Many to one**: Multiple left rows map to one right row (most common)
- **One to many**: One left row maps to multiple right rows
- **One to one**: One left row maps to at most one right row

### Best Practices
- For frequently joined tables: consider pre-joining into denormalized views
- For complex multi-table joins: provide SQL examples demonstrating the correct join pattern
- Specify aliases when multiple joins exist between the same tables

---

## 4. SQL Expressions (Priority 4)

SQL expressions are a structured way to teach Genie about business terms. They have a **separate 200-item limit** (not counted toward the 100-instruction cap).

### Three Types

**Measures** (KPIs and metrics):
```
Name: Total Revenue
Code: SUM(orders.total_amount)
Synonyms: revenue, sales, total sales
Instructions: Use this for any revenue-related aggregation
```

**Filters** (common conditions, must evaluate to boolean):
```
Name: Active Customers
Code: customers.status = 'ACTIVE' AND customers.churn_date IS NULL
Synonyms: current customers, non-churned
Instructions: Apply when users ask about active or current customers
```

**Dimensions** (grouping attributes):
```
Name: Monthly Period
Code: DATE_TRUNC('month', orders.order_date)
Synonyms: by month, monthly
Instructions: Use for monthly aggregation and trending
```

### Best Practices
- Define every KPI/metric as a measure expression
- Create filters for commonly used business conditions
- Add dimensions for standard grouping patterns (monthly, quarterly, by region)
- Include synonyms for each expression (the words users actually use)
- Add instructions explaining when to apply each expression

---

## 5. SQL Examples (Priority 5)

### Limits and Behavior
- **Maximum 100 SQL examples per space**
- Genie uses **vector search** to find relevant examples (doesn't load all 100)
- Quality > quantity: 15-20 diverse, well-crafted examples beat 80 narrow ones

### Example Structure (Always Include)
```sql
-- Question: What are the top 5 products by revenue this quarter?
-- Business Logic: Revenue = SUM(line_total) for completed orders only
SELECT product_name, SUM(line_total) AS revenue
FROM order_lines ol
JOIN products p ON ol.product_id = p.product_id
WHERE ol.order_status = 'COMPLETED'
  AND ol.order_date >= DATE_TRUNC('quarter', CURRENT_DATE())
GROUP BY product_name
ORDER BY revenue DESC
LIMIT 5;
```

### Essential Patterns to Cover
1. **Top N queries** with explicit metric selection
2. **Time-based aggregations** (monthly trends, YoY comparisons)
3. **Filtered aggregations** with business conditions
4. **Multi-table joins** demonstrating correct join usage
5. **Year-over-year / period comparisons** with both current and prior periods

### Parameterized Examples (Trusted Answers)
```sql
-- Question: Show sales for a specific region and time period
-- Parameters: region, start_date, end_date
SELECT product_category, SUM(total_amount) AS revenue
FROM sales
WHERE region = :region
  AND sale_date BETWEEN :start_date AND :end_date
GROUP BY product_category
ORDER BY revenue DESC;
```
When a parameterized example matches exactly, the response is marked "Trusted."

### What NOT to Do
- Don't copy benchmark SQL directly into examples (overfitting)
- Don't create examples for one-off questions (not reusable)
- Don't add variations that only differ in filter values (use entity matching instead)
- Prefer structural patterns that generalize across many user questions

---

## 6. Text Instructions (Priority 6 — Lowest)

### Token Budget
Keep total instruction text under **2,000 characters**:
- General instructions: 300-500 chars (20-30%)
- SQL examples: 800-1,200 chars (50-60%)
- Trusted assets: 400-600 chars (20-30%)

### When to Use Text Instructions
- Business jargon definitions: "Fiscal year starts in February"
- Table disambiguation: "For sales data, always use fact_sales, not fact_sales_archive"
- Clarification triggers: "When users ask about performance without specifying a time range, ask which period"
- Formatting requirements: "Round decimals to 2 places"
- **Only when the behavior cannot be expressed as a SQL expression or example**

### Instruction Template
```
# Purpose
One-line description of what this space answers.

# Business Rules
- [Term] means [definition]
- [Metric] is calculated as [formula]

# Data Guidance
- Use [table_name] for [purpose]
- [Column] valid values: [list]

# Response Format
- Always include [identifiers]
- Round to [N] decimal places
```

### Optimization Techniques
- Bullet points instead of sentences (40% token reduction)
- Abbreviations after first use (25% reduction)
- Table-based mappings for synonyms (35% reduction)
- **Do not repeat information already captured in SQL expressions, entity matching, or column descriptions**

---

## 7. Testing & Benchmarking

### Benchmark Design
- Create 20-50 ground truth questions with expected SQL
- **2-4 phrasings of the same question** to test robustness
- Each benchmark runs as a **new conversation** (no context from prior questions)
- Up to 500 benchmark questions per space

### Rating Criteria
- **Good**: SQL matches, result sets match (including rounding to 4 significant digits)
- **Bad**: Empty results, errors, extra/missing columns, single cell mismatches

### Iteration Strategy
1. Run full benchmark
2. Identify failing questions
3. Diagnose root cause (wrong table? wrong join? wrong filter? wrong metric?)
4. Apply fix using the highest-priority lever that addresses the root cause
5. Re-benchmark to verify improvement without regression
6. Repeat

---

## 8. Common Pitfalls

| Pitfall | Impact | Fix |
|---------|--------|-----|
| Too many tables (>15) | Hallucination | Split into focused spaces |
| Poor column names (abbreviations) | Wrong column selection | Rename or add descriptions |
| Missing entity matching | Filter value mismatches | Enable + curate value lists |
| Undefined joins | Wrong table combinations | Add join specifications |
| Too many text instructions | Reduced effectiveness | Convert to SQL expressions/examples |
| Benchmark SQL copied to examples | Overfitting | Use parameterized patterns instead |
| Stale entity matching values | Missing matches | Refresh when data changes |
| Conflicting instructions | Unpredictable behavior | Audit for consistency |
| Auto-generated descriptions without review | Noise, not signal | Review and edit manually |

---

## 9. Genie Architecture Awareness

- Genie is a **compound system** with multiple planning, retrieval, and SQL-generation components
- It is **nondeterministic**: same question may produce different SQL
- Structural config (SQL expressions, examples, entity matching) produces **more consistent** results than text instructions
- Context is **intelligently filtered** per query — not all metadata is sent every time
- Rate limit: **5 questions per minute** per workspace across all spaces
- Genie can query tables beyond those added to the space (UC permissions control access)
