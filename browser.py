"""
DuckDB Database Browser
View and explore crowd surveillance records
"""

import duckdb
import pandas as pd
from tabulate import tabulate

def browse_database():
    """Browse the DuckDB database and display records"""
    
    # Connect to database
    conn = duckdb.connect("crowd_data.db", read_only=True)
    
    print("\n" + "="*80)
    print("🗄️  CROWD SURVEILLANCE DATABASE BROWSER")
    print("="*80)
    
    # Get total records
    total = conn.execute("SELECT COUNT(*) FROM crowd_metrics").fetchone()[0]
    print(f"\n📊 Total Records: {total}")
    
    if total == 0:
        print("\n⚠️  No records found in database.")
        print("   Run the Streamlit app to start collecting data.")
        conn.close()
        return
    
    # Show table schema
    print("\n📋 Table Schema:")
    print("-" * 40)
    schema = conn.execute("DESCRIBE crowd_metrics").fetchdf()
    print(tabulate(schema, headers='keys', tablefmt='grid', showindex=False))
    
    # Show latest 20 records
    print("\n📈 Latest 20 Records:")
    print("-" * 80)
    
    df = conn.execute("""
        SELECT 
            timestamp,
            people_count,
            ROUND(density, 6) as density,
            flow_direction,
            risk_level
        FROM crowd_metrics 
        ORDER BY timestamp DESC 
        LIMIT 20
    """).fetchdf()
    
    # Format timestamp for display
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
    
    print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))
    
    # Statistics
    print("\n📊 Statistics:")
    print("-" * 40)
    
    stats = conn.execute("""
        SELECT 
            COUNT(*) as total_records,
            AVG(people_count) as avg_count,
            MAX(people_count) as max_count,
            MIN(people_count) as min_count,
            AVG(density) as avg_density,
            MAX(density) as max_density
        FROM crowd_metrics
    """).fetchone()
    
    print(f"  Total Records:    {stats[0]}")
    print(f"  Avg People Count: {stats[1]:.2f}")
    print(f"  Max People Count: {stats[2]}")
    print(f"  Min People Count: {stats[3]}")
    print(f"  Avg Density:      {stats[4]:.6f}")
    print(f"  Max Density:      {stats[5]:.6f}")
    
    # Risk distribution
    print("\n🚨 Risk Level Distribution:")
    print("-" * 40)
    
    risk_dist = conn.execute("""
        SELECT 
            risk_level,
            COUNT(*) as count,
            ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM crowd_metrics), 2) as percentage
        FROM crowd_metrics
        GROUP BY risk_level
        ORDER BY 
            CASE risk_level 
                WHEN 'Risky' THEN 1 
                WHEN 'Average' THEN 2 
                ELSE 3 
            END
    """).fetchdf()
    
    print(tabulate(risk_dist, headers='keys', tablefmt='grid', showindex=False))
    
    # Flow direction distribution
    print("\n🧭 Flow Direction Distribution:")
    print("-" * 40)
    
    flow_dist = conn.execute("""
        SELECT 
            flow_direction,
            COUNT(*) as count
        FROM crowd_metrics
        GROUP BY flow_direction
        ORDER BY count DESC
    """).fetchdf()
    
    print(tabulate(flow_dist, headers='keys', tablefmt='grid', showindex=False))
    
    # Time range
    print("\n⏰ Time Range:")
    print("-" * 40)
    
    time_range = conn.execute("""
        SELECT 
            MIN(timestamp) as first_record,
            MAX(timestamp) as last_record
        FROM crowd_metrics
    """).fetchone()
    
    print(f"  First Record: {time_range[0]}")
    print(f"  Last Record:  {time_range[1]}")
    
    # Export option
    print("\n" + "="*80)
    print("📁 Export Options:")
    print("  Run 'python browser.py export' to export all records to CSV")
    print("="*80 + "\n")
    
    conn.close()


def export_to_csv():
    """Export all records to CSV file"""
    
    conn = duckdb.connect("crowd_data.db", read_only=True)
    
    df = conn.execute("""
        SELECT * FROM crowd_metrics ORDER BY timestamp DESC
    """).fetchdf()
    
    filename = "crowd_data_export.csv"
    df.to_csv(filename, index=False)
    
    print(f"\n✅ Exported {len(df)} records to {filename}")
    
    conn.close()


def search_records(search_term: str):
    """Search records by risk level or flow direction"""
    
    conn = duckdb.connect("crowd_data.db", read_only=True)
    
    df = conn.execute(f"""
        SELECT 
            timestamp,
            people_count,
            ROUND(density, 6) as density,
            flow_direction,
            risk_level
        FROM crowd_metrics 
        WHERE risk_level ILIKE '%{search_term}%' 
           OR flow_direction ILIKE '%{search_term}%'
        ORDER BY timestamp DESC 
        LIMIT 50
    """).fetchdf()
    
    if len(df) > 0:
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
        print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))
    else:
        print(f"No records found matching '{search_term}'")
    
    conn.close()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "export":
            export_to_csv()
        elif sys.argv[1] == "search" and len(sys.argv) > 2:
            search_records(sys.argv[2])
        else:
            print("Usage:")
            print("  python browser.py           - View database records")
            print("  python browser.py export    - Export to CSV")
            print("  python browser.py search <term> - Search records")
    else:
        browse_database()
