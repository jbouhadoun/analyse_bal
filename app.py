#!/usr/bin/env python3
"""
BAL Analysis Dashboard v10
Complete dashboard with filters, statistics, visualizations + Shared IDs detection
"""

import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

DB_PATH = Path('data/bal_analysis_v2.db')

st.set_page_config(
    page_title="BAL Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# DATABASE CONNECTION
# ============================================================================

@st.cache_resource
def get_db_connection():
    """Get database connection."""
    if not DB_PATH.exists():
        st.error(f"Database not found: {DB_PATH}")
        st.info("Please run `python scripts/rebuild_database_from_json.py` first")
        st.stop()
    return sqlite3.connect(DB_PATH, check_same_thread=False)

# ============================================================================
# DATA LOADING
# ============================================================================

@st.cache_data(ttl=60)
def load_communes(_conn, filters=None):
    """Load communes with optional filters."""
    
    query = """
        SELECT 
            code_commune, nom_commune, code_departement,
            producer, mandataire, organisation,
            total_addresses, 
            CASE 
                WHEN change_rate > 100 THEN 100 
                ELSE change_rate 
            END as change_rate,
            status, status_freshness,
            has_uid_adresse, has_id_ban_commune, has_id_ban_toponyme, has_id_ban_adresse,
            days_since_last_update, current_version,
            volatility_class, volatility, validation_valid,
            recent_stability_score, avg_stability_score, is_improving,
            producer_changes_count, mandataire_changes_count, organisation_changes_count,
            last_revision_date, has_bal, has_shared_ids, shared_ids_count
        FROM communes
        WHERE 1=1
    """
    
    params = []
    
    if filters:
        # BAL status filter
        if 'has_bal' in filters:
            query += " AND has_bal = ?"
            params.append(1 if filters['has_bal'] else 0)
        
        if filters.get('departements'):
            placeholders = ','.join(['?' for _ in filters['departements']])
            query += f" AND code_departement IN ({placeholders})"
            params.extend(filters['departements'])
        
        if filters.get('producers'):
            placeholders = ','.join(['?' for _ in filters['producers']])
            query += f" AND producer IN ({placeholders})"
            params.extend(filters['producers'])
        
        if filters.get('mandataires'):
            placeholders = ','.join(['?' for _ in filters['mandataires']])
            query += f" AND mandataire IN ({placeholders})"
            params.extend(filters['mandataires'])
        
        if filters.get('organisations'):
            placeholders = ','.join(['?' for _ in filters['organisations']])
            query += f" AND organisation IN ({placeholders})"
            params.extend(filters['organisations'])
        
        if filters.get('statuses'):
            placeholders = ','.join(['?' for _ in filters['statuses']])
            query += f" AND status IN ({placeholders})"
            params.extend(filters['statuses'])
        
        if filters.get('freshness'):
            placeholders = ','.join(['?' for _ in filters['freshness']])
            query += f" AND status_freshness IN ({placeholders})"
            params.extend(filters['freshness'])
        
        if filters.get('with_uid_only'):
            query += " AND has_uid_adresse = 1"
        
        if filters.get('without_id_adresse'):
            query += " AND has_id_ban_adresse = 0"
        
        if filters.get('with_shared_ids'):
            query += " AND has_shared_ids = 1"
        
        if filters.get('with_producer_changes'):
            query += " AND producer_changes_count >= 1"
        
        if filters.get('with_mandataire_changes'):
            query += " AND mandataire_changes_count >= 1"
        
        if filters.get('with_organisation_changes'):
            query += " AND organisation_changes_count >= 1"
    
    query += " ORDER BY change_rate DESC"
    
    df = pd.read_sql_query(query, _conn, params=params if params else None)
    return df

@st.cache_data(ttl=60)
def load_shared_ids(_conn, code_commune=None):
    """Load shared IDs details."""
    if code_commune:
        query = """
            SELECT id_value, id_type, communes_list, nb_communes
            FROM shared_ids
            WHERE communes_list LIKE ?
            ORDER BY nb_communes DESC
        """
        return pd.read_sql_query(query, _conn, params=[f'%{code_commune}%'])
    else:
        query = "SELECT * FROM shared_ids ORDER BY nb_communes DESC LIMIT 100"
        return pd.read_sql_query(query, _conn)

@st.cache_data(ttl=60)
def load_departements(_conn):
    """Load department statistics."""
    query = "SELECT * FROM departements ORDER BY code_departement"
    return pd.read_sql_query(query, _conn)

@st.cache_data(ttl=60)
def load_producteurs(_conn):
    """Load producer statistics."""
    query = "SELECT * FROM producteurs ORDER BY total_communes DESC"
    return pd.read_sql_query(query, _conn)

@st.cache_data(ttl=60)
def load_mandataires(_conn):
    """Load mandataire statistics."""
    query = "SELECT * FROM mandataires ORDER BY total_communes DESC"
    return pd.read_sql_query(query, _conn)

@st.cache_data(ttl=60)
def load_organisations(_conn):
    """Load organisation statistics."""
    query = "SELECT * FROM organisations ORDER BY total_communes DESC"
    return pd.read_sql_query(query, _conn)

@st.cache_data(ttl=60)
def get_filter_options(_conn, current_filters=None):
    """Get unique values for filters based on current selections (cascading filters)."""
    
    cursor = _conn.cursor()
    
    # Build WHERE clause based on current filters
    where_clauses = []
    params = []
    
    # Add BAL status filter if specified
    if current_filters and 'has_bal' in current_filters:
        where_clauses.append("has_bal = ?")
        params.append(1 if current_filters['has_bal'] else 0)
    
    if current_filters:
        if current_filters.get('departements'):
            placeholders = ','.join(['?' for _ in current_filters['departements']])
            where_clauses.append(f"code_departement IN ({placeholders})")
            params.extend(current_filters['departements'])
        
        if current_filters.get('producers'):
            placeholders = ','.join(['?' for _ in current_filters['producers']])
            where_clauses.append(f"producer IN ({placeholders})")
            params.extend(current_filters['producers'])
        
        if current_filters.get('mandataires'):
            placeholders = ','.join(['?' for _ in current_filters['mandataires']])
            where_clauses.append(f"mandataire IN ({placeholders})")
            params.extend(current_filters['mandataires'])
        
        if current_filters.get('organisations'):
            placeholders = ','.join(['?' for _ in current_filters['organisations']])
            where_clauses.append(f"organisation IN ({placeholders})")
            params.extend(current_filters['organisations'])
    
    # Build WHERE clause (with fallback if empty)
    where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"
    
    # Get departments
    cursor.execute(f"SELECT DISTINCT code_departement FROM communes WHERE {where_sql} ORDER BY code_departement", params)
    departements = [row[0] for row in cursor.fetchall() if row[0]]
    
    # Get producers
    cursor.execute(f"SELECT DISTINCT producer FROM communes WHERE {where_sql} AND producer IS NOT NULL ORDER BY producer", params)
    producers = [row[0] for row in cursor.fetchall()]
    
    # Get mandataires
    cursor.execute(f"SELECT DISTINCT mandataire FROM communes WHERE {where_sql} AND mandataire IS NOT NULL ORDER BY mandataire", params)
    mandataires = [row[0] for row in cursor.fetchall()]
    
    # Get organisations
    cursor.execute(f"SELECT DISTINCT organisation FROM communes WHERE {where_sql} AND organisation IS NOT NULL ORDER BY organisation", params)
    organisations = [row[0] for row in cursor.fetchall()]
    
    return {
        'departements': departements,
        'producers': producers,
        'mandataires': mandataires,
        'organisations': organisations
    }

# ============================================================================
# HEADER
# ============================================================================

def show_header():
    """Display page header."""
    st.title("BAL Analysis Dashboard")
    st.markdown("**Analyse de la stabilité des Bases Adresses Locales**")
    st.markdown("---")

# ============================================================================
# FILTERS SIDEBAR
# ============================================================================

def show_filters(conn):
    """Show filters in sidebar with cascading logic (no icons, clean design)."""
    
    st.sidebar.header("Filtres")
    
    filters = {}
    
    # BAL status filter - SIMPLIFIED TO 2 OPTIONS
    st.sidebar.subheader("Statut BAL")
    bal_status = st.sidebar.radio(
        "Sélectionner",
        options=["Avec BAL", "Sans BAL"],
        index=0,  # Default: Avec BAL
        label_visibility="collapsed"
    )
    
    # Set has_bal filter
    if bal_status == "Avec BAL":
        filters['has_bal'] = True
    else:  # Sans BAL
        filters['has_bal'] = False
    
    st.sidebar.markdown("---")
    
    # ALL OTHER FILTERS: Only enabled for "Avec BAL"
    if bal_status == "Avec BAL":
        # Get initial filter options
        filter_options = get_filter_options(conn, filters)
        
        # Departments
        st.sidebar.subheader("Département")
        selected_depts = st.sidebar.multiselect(
            "Sélectionner département(s)",
            options=filter_options['departements'],
            default=None,
            label_visibility="collapsed"
        )
        if selected_depts:
            filters['departements'] = selected_depts
        
        # Update filter options based on department selection
        if selected_depts:
            filter_options = get_filter_options(conn, filters)
        
        # Producers
        st.sidebar.subheader("Producteur")
        selected_producers = st.sidebar.multiselect(
            "Sélectionner producteur(s)",
            options=filter_options['producers'],
            default=None,
            label_visibility="collapsed"
        )
        if selected_producers:
            filters['producers'] = selected_producers
        
        # Update filter options based on producer selection
        if selected_producers:
            filter_options = get_filter_options(conn, filters)
        
        # Mandataires
        st.sidebar.subheader("Mandataire")
        selected_mandataires = st.sidebar.multiselect(
            "Sélectionner mandataire(s)",
            options=filter_options['mandataires'],
            default=None,
            label_visibility="collapsed"
        )
        if selected_mandataires:
            filters['mandataires'] = selected_mandataires
        
        # Update filter options based on mandataire selection
        if selected_mandataires:
            filter_options = get_filter_options(conn, filters)
        
        # Organisations
        st.sidebar.subheader("Organisation")
        selected_organisations = st.sidebar.multiselect(
            "Sélectionner organisation(s)",
            options=filter_options['organisations'],
            default=None,
            label_visibility="collapsed"
        )
        if selected_organisations:
            filters['organisations'] = selected_organisations
        
        st.sidebar.markdown("---")
        
        # Status
        st.sidebar.subheader("Statut Changements")
        status_options = {
            'BON': st.sidebar.checkbox("BON (<25%)", value=True),
            'WARNING': st.sidebar.checkbox("WARNING (25-80%)", value=True),
            'CRITIQUE': st.sidebar.checkbox("CRITIQUE (>80%)", value=True)
        }
        selected_statuses = [k for k, v in status_options.items() if v]
        if selected_statuses:
            filters['statuses'] = selected_statuses
        
        # Freshness
        st.sidebar.subheader("Fraîcheur")
        freshness_options = {
            'ACTIVE': st.sidebar.checkbox("ACTIVE (<1 mois)", value=True),
            'RALENTIE': st.sidebar.checkbox("RALENTIE (1-6 mois)", value=True),
            'INACTIVE': st.sidebar.checkbox("INACTIVE (6 mois-2 ans)", value=True),
            'MORTE': st.sidebar.checkbox("MORTE (>2 ans)", value=True)
        }
        selected_freshness = [k for k, v in freshness_options.items() if v]
        if selected_freshness:
            filters['freshness'] = selected_freshness
        
        st.sidebar.markdown("---")
        
        # Identifiers (simplified + SHARED IDS)
        st.sidebar.subheader("Identifiants")
        filters['without_id_adresse'] = st.sidebar.checkbox("Sans identifiants")
        filters['with_uid_only'] = st.sidebar.checkbox("Avec uid_adresse")
        filters['with_shared_ids'] = st.sidebar.checkbox("Avec IDs partagés")
        
        st.sidebar.markdown("---")
        
        # Changes filters
        st.sidebar.subheader("Changements")
        filters['with_producer_changes'] = st.sidebar.checkbox("Avec changement producteur (≥1)")
        filters['with_mandataire_changes'] = st.sidebar.checkbox("Avec changement mandataire (≥1)")
        filters['with_organisation_changes'] = st.sidebar.checkbox("Avec changement organisation (≥1)")
        
        st.sidebar.markdown("---")
        
        # Reset button
        if st.sidebar.button("Réinitialiser les filtres"):
            st.rerun()
    
    else:  # Sans BAL
        # No other filters - just show the list
        st.sidebar.info("Affichage liste des communes sans BAL")
        st.sidebar.markdown("Pas de filtres disponibles pour les communes sans BAL")
    
    return filters

# ============================================================================
# OVERVIEW STATISTICS
# ============================================================================

def show_overview(df):
    """Show overview statistics."""
    
    st.header("Vue d'ensemble")
    
    total_communes = len(df)
    
    if total_communes == 0:
        st.warning("Aucune commune sélectionnée")
        return
    
    # Status counts
    status_counts = df['status'].value_counts()
    nb_bon = status_counts.get('BON', 0)
    nb_warning = status_counts.get('WARNING', 0)
    nb_critique = status_counts.get('CRITIQUE', 0)
    
    # Freshness counts
    freshness_counts = df['status_freshness'].value_counts()
    nb_morte = freshness_counts.get('MORTE', 0)
    
    # Identifiers
    nb_with_uid = df['has_uid_adresse'].sum()
    nb_without_id_adresse = (df['has_id_ban_adresse'] == 0).sum()
    
    # Shared IDs
    nb_with_shared_ids = (df['has_shared_ids'] == 1).sum() if 'has_shared_ids' in df.columns else 0
    
    # Average change rate
    avg_change_rate = df['change_rate'].mean() if not df['change_rate'].isna().all() else 0
    
    # Display metrics (first row)
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Total Communes",
            f"{total_communes:,}",
            help="Communes sélectionnées selon les filtres"
        )
    
    with col2:
        st.metric(
            "BON",
            f"{nb_bon:,}",
            f"{nb_bon/total_communes*100:.1f}%" if total_communes > 0 else "0%",
            help="Moins de 25% de changements"
        )
    
    with col3:
        st.metric(
            "WARNING",
            f"{nb_warning:,}",
            f"{nb_warning/total_communes*100:.1f}%" if total_communes > 0 else "0%",
            help="Entre 25% et 80% de changements"
        )
    
    with col4:
        st.metric(
            "CRITIQUE",
            f"{nb_critique:,}",
            f"{nb_critique/total_communes*100:.1f}%" if total_communes > 0 else "0%",
            help="Plus de 80% de changements"
        )
    
    with col5:
        st.metric(
            "MORTE (>2 ans)",
            f"{nb_morte:,}",
            f"{nb_morte/total_communes*100:.1f}%" if total_communes > 0 else "0%",
            help="Pas de mise à jour depuis plus de 2 ans"
        )
    
    st.markdown("---")
    
    # Second row of metrics (WITH SHARED IDS)
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Taux changement moyen",
            f"{avg_change_rate:.1f}%",
            help="Pourcentage moyen de changements d'identifiants (max 100%)"
        )
    
    with col2:
        st.metric(
            "Avec uid_adresse",
            f"{nb_with_uid:,}",
            f"{nb_with_uid/total_communes*100:.1f}%" if total_communes > 0 else "0%",
            help="Communes utilisant uid_adresse (BAL 1.4)"
        )
    
    with col3:
        st.metric(
            "Sans identifiants",
            f"{nb_without_id_adresse:,}",
            f"{nb_without_id_adresse/total_communes*100:.1f}%" if total_communes > 0 else "0%",
            help="Communes sans id_ban_adresse"
        )
    
    with col4:
        st.metric(
            "Avec IDs partagés",
            f"{nb_with_shared_ids:,}",
            f"{nb_with_shared_ids/total_communes*100:.1f}%" if total_communes > 0 else "0%",
            help="Communes avec identifiants partagés avec d'autres communes"
        )
    
    with col5:
        total_addresses = df['total_addresses'].sum()
        st.metric(
            "Total adresses",
            f"{int(total_addresses):,}",
            help="Nombre total d'adresses"
        )

# ============================================================================
# CHARTS
# ============================================================================

def show_charts(df):
    """Show various charts."""
    
    st.header("Graphiques")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Status distribution
        st.subheader("Distribution des statuts")
        status_counts = df['status'].value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=status_counts.index,
            values=status_counts.values,
            marker_colors=['#28a745', '#ffc107', '#dc3545'],
            hole=0.4
        )])
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Freshness distribution
        st.subheader("Distribution fraîcheur")
        freshness_counts = df['status_freshness'].value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=freshness_counts.index,
            values=freshness_counts.values,
            marker_colors=['#28a745', '#ffc107', '#fd7e14', '#dc3545'],
            hole=0.4
        )])
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Change rate distribution
    st.subheader("Distribution des taux de changement")
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df['change_rate'].dropna(),
        nbinsx=50,
        marker_color='#007bff'
    ))
    fig.update_layout(
        xaxis_title="Taux de changement (%)",
        yaxis_title="Nombre de communes",
        height=400
    )
    
    # Add threshold lines
    fig.add_vline(x=25, line_dash="dash", line_color="orange", annotation_text="WARNING (25%)")
    fig.add_vline(x=80, line_dash="dash", line_color="red", annotation_text="CRITIQUE (80%)")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Top departments by change rate
    st.subheader("Top 10 départements par taux de changement")
    
    dept_stats = df.groupby('code_departement').agg({
        'change_rate': 'mean',
        'code_commune': 'count'
    }).reset_index()
    dept_stats.columns = ['Département', 'Taux changement moyen', 'Nb communes']
    dept_stats = dept_stats.sort_values('Taux changement moyen', ascending=False).head(10)
    
    fig = px.bar(
        dept_stats,
        x='Département',
        y='Taux changement moyen',
        color='Taux changement moyen',
        color_continuous_scale=['green', 'yellow', 'red'],
        text='Taux changement moyen'
    )
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# DETAILED TABLE
# ============================================================================

def show_detailed_table(df, conn):
    """Show detailed table with sorting and pagination + Shared IDs details."""
    
    st.header("Liste détaillée des communes")
    
    # Sort options
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        sort_by = st.selectbox(
            "Trier par",
            options=[
                'change_rate', 'nom_commune', 'code_commune', 'total_addresses', 
                'days_since_last_update', 'recent_stability_score', 'volatility', 
                'producer_changes_count', 'mandataire_changes_count', 'organisation_changes_count',
                'shared_ids_count'
            ],
            format_func=lambda x: {
                'change_rate': 'Taux de changement',
                'nom_commune': 'Nom commune',
                'code_commune': 'Code commune',
                'total_addresses': 'Nombre d\'adresses',
                'days_since_last_update': 'Jours depuis MAJ',
                'recent_stability_score': 'Stabilité récente',
                'volatility': 'Volatilité',
                'producer_changes_count': 'Changements producteur',
                'mandataire_changes_count': 'Changements mandataire',
                'organisation_changes_count': 'Changements organisation',
                'shared_ids_count': 'IDs partagés'
            }[x],
            index=0
        )
    
    with col2:
        sort_order = st.selectbox(
            "Ordre",
            options=['desc', 'asc'],
            format_func=lambda x: 'Décroissant' if x == 'desc' else 'Croissant'
        )
    
    with col3:
        page_size = st.selectbox(
            "Lignes par page",
            options=[10, 25, 50, 100],
            index=1
        )
    
    # Sort dataframe
    df_sorted = df.sort_values(sort_by, ascending=(sort_order == 'asc'))
    
    # Pagination
    total_rows = len(df_sorted)
    total_pages = max(1, (total_rows + page_size - 1) // page_size)
    
    page = st.number_input(
        f"Page (1-{total_pages})",
        min_value=1,
        max_value=total_pages,
        value=1,
        step=1
    )
    
    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, total_rows)
    
    df_page = df_sorted.iloc[start_idx:end_idx]
    
    # Prepare display columns
    display_cols = [
        'code_commune', 'nom_commune', 'code_departement',
        'producer', 'mandataire', 'organisation',
        'change_rate', 'status', 'status_freshness',
        'total_addresses',
        'recent_stability_score', 'volatility', 'is_improving',
        'producer_changes_count', 'mandataire_changes_count', 'organisation_changes_count',
        'days_since_last_update'
    ]
    
    # Add shared_ids_count if available
    if 'shared_ids_count' in df_page.columns:
        display_cols.append('shared_ids_count')
    
    # Display table with all important columns
    st.dataframe(
        df_page[display_cols].rename(columns={
            'code_commune': 'Code',
            'nom_commune': 'Commune',
            'code_departement': 'Dept',
            'producer': 'Producteur',
            'mandataire': 'Mandataire',
            'organisation': 'Organisation',
            'change_rate': 'Tx change (%)',
            'status': 'Statut',
            'status_freshness': 'Fraîcheur',
            'total_addresses': 'Nb adresses',
            'recent_stability_score': 'Stabilité récente',
            'volatility': 'Volatilité',
            'is_improving': 'En amélioration',
            'producer_changes_count': 'Ch. producteur',
            'mandataire_changes_count': 'Ch. mandataire',
            'organisation_changes_count': 'Ch. organisation',
            'days_since_last_update': 'Jours MAJ',
            'shared_ids_count': 'IDs partagés'
        }),
        use_container_width=True,
        height=400
    )
    
    # Show shared IDs details for communes with shared IDs
    if 'has_shared_ids' in df_page.columns and (df_page['has_shared_ids'] == 1).any():
        st.markdown("---")
        st.subheader("Détails des IDs partagés")
        
        communes_with_shared = df_page[df_page['has_shared_ids'] == 1]
        
        selected_commune = st.selectbox(
            "Sélectionner une commune pour voir les détails",
            options=communes_with_shared['code_commune'].tolist(),
            format_func=lambda x: f"{x} - {communes_with_shared[communes_with_shared['code_commune']==x]['nom_commune'].iloc[0]}"
        )
        
        if selected_commune:
            shared_ids_df = load_shared_ids(conn, selected_commune)
            
            if not shared_ids_df.empty:
                st.write(f"**{len(shared_ids_df)} identifiant(s) partagé(s) détecté(s)**")
                
                for _, row in shared_ids_df.iterrows():
                    with st.expander(f"**{row['id_type']}** : `{row['id_value'][:50]}...`"):
                        st.write(f"**Partagé avec {row['nb_communes']} commune(s)** :")
                        communes_list = row['communes_list'].split(',')
                        st.write(", ".join(communes_list))
            else:
                st.info("Aucun détail disponible")
    
    st.caption(f"Affichage : {start_idx + 1}-{end_idx} sur {total_rows:,} communes")
    
    # Export button
    csv = df_sorted.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Télécharger CSV complet",
        data=csv,
        file_name=f"bal_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

# ============================================================================
# DEPARTMENTS TAB
# ============================================================================

def show_departments_tab(conn):
    """Show department statistics."""
    
    st.header("Statistiques par département")
    
    df_dept = load_departements(conn)
    
    if not df_dept.empty:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Départements", len(df_dept))
        
        with col2:
            avg_coverage = df_dept['coverage_rate'].mean()
            st.metric("Couverture moyenne", f"{avg_coverage:.1f}%")
        
        with col3:
            total_critique = df_dept['nb_critique'].sum()
            st.metric("Total communes critiques", f"{int(total_critique):,}")
        
        with col4:
            total_morte = df_dept['nb_morte'].sum()
            st.metric("Total communes mortes", f"{int(total_morte):,}")
        
        st.markdown("---")
        
        # Detailed table
        st.subheader("Détail par département")
        
        df_display = df_dept[[
            'code_departement', 'total_communes', 'communes_avec_bal', 'communes_sans_bal',
            'avg_change_rate', 'nb_bon', 'nb_warning', 'nb_critique',
            'nb_active', 'nb_morte'
        ]].rename(columns={
            'code_departement': 'Département',
            'total_communes': 'Total',
            'communes_avec_bal': 'Avec BAL',
            'communes_sans_bal': 'Sans BAL',
            'avg_change_rate': 'Tx change moy (%)',
            'nb_bon': 'BON',
            'nb_warning': 'WARNING',
            'nb_critique': 'CRITIQUE',
            'nb_active': 'ACTIVE',
            'nb_morte': 'MORTE'
        })
        
        st.dataframe(df_display, use_container_width=True, height=600)
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Top 10 by change rate
            st.subheader("Top 10 taux de changement")
            df_top = df_dept.nlargest(10, 'avg_change_rate')
            
            fig = px.bar(
                df_top,
                x='code_departement',
                y='avg_change_rate',
                color='avg_change_rate',
                color_continuous_scale=['green', 'yellow', 'red']
            )
            fig.update_layout(xaxis_title="Département", yaxis_title="Taux moyen (%)")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Top 10 by critique count
            st.subheader("Top 10 communes critiques")
            df_top = df_dept.nlargest(10, 'nb_critique')
            
            fig = px.bar(
                df_top,
                x='code_departement',
                y='nb_critique',
                color='nb_critique',
                color_continuous_scale='Reds'
            )
            fig.update_layout(xaxis_title="Département", yaxis_title="Nb communes critiques")
            st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PRODUCERS TAB
# ============================================================================

def show_producers_tab(conn):
    """Show producer statistics."""
    
    st.header("Statistiques par producteur")
    
    df_prod = load_producteurs(conn)
    
    if not df_prod.empty:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Producteurs", len(df_prod))
        
        with col2:
            best_prod = df_prod.nsmallest(1, 'avg_change_rate').iloc[0]
            st.metric(
                "Meilleur producteur",
                best_prod['producer'],
                f"{best_prod['avg_change_rate']:.1f}%"
            )
        
        with col3:
            worst_prod = df_prod.nlargest(1, 'avg_change_rate').iloc[0]
            st.metric(
                "Pire producteur",
                worst_prod['producer'],
                f"{worst_prod['avg_change_rate']:.1f}%"
            )
        
        with col4:
            total_critique = df_prod['nb_critique'].sum()
            st.metric("Total communes critiques", f"{int(total_critique):,}")
        
        st.markdown("---")
        
        # Detailed table
        st.subheader("Détail par producteur")
        
        df_display = df_prod[[
            'producer', 'total_communes', 'avg_change_rate',
            'nb_bon', 'nb_warning', 'nb_critique',
            'nb_active', 'nb_morte',
            'nb_with_uid_adresse'
        ]].rename(columns={
            'producer': 'Producteur',
            'total_communes': 'Nb communes',
            'avg_change_rate': 'Tx change moy (%)',
            'nb_bon': 'BON',
            'nb_warning': 'WARNING',
            'nb_critique': 'CRITIQUE',
            'nb_active': 'ACTIVE',
            'nb_morte': 'MORTE',
            'nb_with_uid_adresse': 'Avec uid_adresse'
        })
        
        st.dataframe(df_display, use_container_width=True, height=400)
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Communes by producer
            st.subheader("Répartition des communes")
            
            fig = px.pie(
                df_prod,
                values='total_communes',
                names='producer',
                title="Nombre de communes par producteur"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Average change rate by producer
            st.subheader("Taux de changement moyen")
            
            fig = px.bar(
                df_prod.sort_values('avg_change_rate', ascending=False),
                x='producer',
                y='avg_change_rate',
                color='avg_change_rate',
                color_continuous_scale=['green', 'yellow', 'red']
            )
            fig.update_layout(xaxis_title="Producteur", yaxis_title="Taux moyen (%)")
            st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main application."""
    
    # Get database connection
    conn = get_db_connection()
    
    # Show header
    show_header()
    
    # Show filters in sidebar (with cascading logic)
    filters = show_filters(conn)
    
    # Load data with filters
    df = load_communes(conn, filters)
    
    if df.empty:
        st.warning("Aucune commune ne correspond aux filtres sélectionnés")
        return
    
    # Check if Sans BAL mode
    is_sans_bal_mode = filters.get('has_bal') == False
    
    if is_sans_bal_mode:
        # SANS BAL MODE: Show ONLY the detailed list
        st.info("Mode Sans BAL : Affichage de la liste des communes uniquement")
        show_detailed_table(df, conn)
    
    else:
        # AVEC BAL MODE: Show all tabs
        # Create tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "Vue d'ensemble",
            "Liste détaillée",
            "Départements",
            "Producteurs",
            "Mandataires",
            "Organisations"
        ])
        
        with tab1:
            show_overview(df)
            st.markdown("---")
            show_charts(df)
        
        with tab2:
            show_detailed_table(df, conn)
        
        with tab3:
            show_departments_tab(conn)
        
        with tab4:
            show_producers_tab(conn)
        
        with tab5:
            st.header("Statistiques par mandataire")
            df_mand = load_mandataires(conn)
            
            if not df_mand.empty:
                df_display = df_mand[[
                    'mandataire', 'total_communes', 'avg_change_rate',
                    'nb_bon', 'nb_warning', 'nb_critique'
                ]].rename(columns={
                    'mandataire': 'Mandataire',
                    'total_communes': 'Nb communes',
                    'avg_change_rate': 'Tx change moy (%)',
                    'nb_bon': 'BON',
                    'nb_warning': 'WARNING',
                    'nb_critique': 'CRITIQUE'
                })
                
                st.dataframe(df_display, use_container_width=True, height=400)
        
        with tab6:
            st.header("Statistiques par organisation")
            df_org = load_organisations(conn)
            
            if not df_org.empty:
                # Metrics summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Organisations", len(df_org))
                with col2:
                    st.metric("Total communes", df_org['total_communes'].sum())
                with col3:
                    avg_change = df_org['avg_change_rate'].mean()
                    st.metric("Tx change moyen", f"{avg_change:.1f}%")
                
                st.markdown("---")
                
                # Detailed table
                df_display = df_org[[
                    'organisation', 'total_communes', 'avg_change_rate',
                    'nb_bon', 'nb_warning', 'nb_critique',
                    'nb_active', 'nb_ralentie', 'nb_inactive', 'nb_morte'
                ]].rename(columns={
                    'organisation': 'Organisation',
                    'total_communes': 'Nb communes',
                    'avg_change_rate': 'Tx change moy (%)',
                    'nb_bon': 'BON',
                    'nb_warning': 'WARNING',
                    'nb_critique': 'CRITIQUE',
                    'nb_active': 'ACTIVE',
                    'nb_ralentie': 'RALENTIE',
                    'nb_inactive': 'INACTIVE',
                    'nb_morte': 'MORTE'
                })
                
                st.dataframe(df_display, use_container_width=True, height=400)
                
                # Charts
                st.subheader("Répartition des communes par organisation")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Pie chart
                    fig = px.pie(
                        df_org, 
                        values='total_communes', 
                        names='organisation',
                        title='Répartition'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Bar chart statuts
                    fig = go.Figure(data=[
                        go.Bar(name='BON', x=df_org['organisation'], y=df_org['nb_bon']),
                        go.Bar(name='WARNING', x=df_org['organisation'], y=df_org['nb_warning']),
                        go.Bar(name='CRITIQUE', x=df_org['organisation'], y=df_org['nb_critique'])
                    ])
                    fig.update_layout(title='Statuts par organisation', barmode='stack')
                    st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.caption(f"Total communes affichées : {len(df):,}")
    st.caption(f"Dernière mise à jour : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()