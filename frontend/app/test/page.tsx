"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Activity, UploadCloud, Scan, Microchip, Layers, ArrowLeft } from "lucide-react";
import Link from "next/link";
import styles from "./test.module.css";

export default function TestSystem() {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState<any>(null);

  const handleAnalyze = () => {
    setIsAnalyzing(true);
    // Simulate API delay
    setTimeout(() => {
      setIsAnalyzing(false);
      setResults({
        diagnosis: "Renal Tumor Detected",
        confidence: "98.4%",
        volume: "125.4 cm³",
        diameter: "4.2 cm",
        stage: "Stage I (T1)",
        localization: "Left Kidney - Lower Cortex Region"
      });
    }, 2500);
  };

  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <Link href="/" style={{display: 'inline-flex', alignItems: 'center', gap:'0.5rem', color:'var(--text-muted)', marginBottom:'1rem'}}>
          <ArrowLeft size={16} /> Back to Home
        </Link>
        <h1 className={styles.title}><Microchip size={32} color="var(--accent-primary)" /> AI Inference Dashboard</h1>
        <p className={styles.subtitle}>Upload patient CT and MRI scans for multi-modal analysis</p>
      </header>

      <div className={styles.layout}>
        {/* Sidebar */}
        <div className={styles.sidebar}>
          <div className={styles.card}>
            <h2 className={styles.cardTitle}><Activity size={20} /> Patient Data Input</h2>
            
            <div className={styles.inputGroup}>
              <label className={styles.label}>Patient Scan ID</label>
              <input type="text" className={styles.input} placeholder="e.g. PT-10024" />
            </div>

            <div className={styles.inputGroup}>
              <label className={styles.label}>T1-Weighted MRI Scan</label>
              <div className={styles.uploadZone}>
                <UploadCloud className={styles.uploadIcon} size={28} />
                <div className={styles.uploadText}>Click or drag to upload MRI</div>
              </div>
            </div>

            <div className={styles.inputGroup}>
              <label className={styles.label}>Contrast CT Scan</label>
              <div className={styles.uploadZone}>
                <UploadCloud className={styles.uploadIcon} size={28} />
                <div className={styles.uploadText}>Click or drag to upload CT</div>
              </div>
            </div>

            <button 
              className={styles.analyzeBtn} 
              onClick={handleAnalyze}
              disabled={isAnalyzing}
            >
              <Scan size={20} />
              {isAnalyzing ? "Processing Fusion Model..." : "Analyze Scans"}
            </button>
          </div>
        </div>

        {/* Main Content View */}
        <div className={styles.mainContent}>
          <AnimatePresence mode="wait">
            {!results && !isAnalyzing ? (
              <motion.div 
                key="waiting"
                className={styles.card} 
                style={{textAlign:'center', padding:'6rem 2rem', height:'100%', display:'flex', flexDirection:'column', justifyContent:'center', alignItems:'center'}}
                initial={{opacity:0}} animate={{opacity:1}} exit={{opacity:0}}
              >
                <Layers size={48} color="var(--border-strong)" style={{marginBottom:'1.5rem'}} />
                <h3 style={{fontSize:'1.5rem', marginBottom:'0.5rem'}}>Waiting for Input</h3>
                <p style={{color:'var(--text-muted)'}}>Upload coregistered scans and click analyze to view the diagnostic report.</p>
              </motion.div>
            ) : isAnalyzing ? (
              <motion.div 
                key="analyzing"
                className={styles.card} 
                style={{textAlign:'center', padding:'6rem 2rem', height:'100%', display:'flex', flexDirection:'column', justifyContent:'center', alignItems:'center'}}
                initial={{opacity:0}} animate={{opacity:1}} exit={{opacity:0}}
              >
                <div style={{position:'relative', width:'80px', height:'80px', marginBottom:'2rem'}}>
                  <motion.div
                    animate={{ rotate: 360 }}
                    transition={{ repeat: Infinity, duration: 2, ease: "linear" }}
                    style={{ position:'absolute', inset:0, border:'3px solid transparent', borderTopColor:'var(--accent-primary)', borderRadius:'50%' }}
                  />
                  <motion.div
                    animate={{ rotate: -360 }}
                    transition={{ repeat: Infinity, duration: 3, ease: "linear" }}
                    style={{ position:'absolute', inset:'10px', border:'3px solid transparent', borderTopColor:'var(--accent-secondary)', borderRadius:'50%' }}
                  />
                  <Scan size={32} style={{position:'absolute', top:'50%', left:'50%', transform:'translate(-50%, -50%)', color:'var(--text-main)'}} />
                </div>
                <h3 style={{fontSize:'1.5rem', marginBottom:'0.5rem'}}>Running U-Net Inference...</h3>
                <p style={{color:'var(--accent-primary)'}}>Aggregating spatial & contrast gradients</p>
              </motion.div>
            ) : (
              <motion.div 
                key="results"
                initial={{opacity:0, y:20}} 
                animate={{opacity:1, y:0}} 
                transition={{duration:0.5}}
                style={{display:'flex', flexDirection:'column', gap:'2rem'}}
              >
                <div className={styles.metricsGrid}>
                  <div className={styles.metricCard} style={{borderColor: 'var(--error)', background: 'rgba(239, 68, 68, 0.05)'}}>
                    <span className={styles.metricLabel}>Diagnosis</span>
                    <span className={styles.metricValue} style={{color:'var(--error)'}}>{results.diagnosis}</span>
                  </div>
                  <div className={styles.metricCard}>
                    <span className={styles.metricLabel}>AI Confidence</span>
                    <span className={styles.metricValue} style={{color:'var(--success)'}}>{results.confidence}</span>
                  </div>
                  <div className={styles.metricCard}>
                    <span className={styles.metricLabel}>Tumor Volume</span>
                    <span className={styles.metricValue}>{results.volume}</span>
                  </div>
                  <div className={styles.metricCard}>
                    <span className={styles.metricLabel}>Max Diameter</span>
                    <span className={styles.metricValue}>{results.diameter}</span>
                  </div>
                  <div className={styles.metricCard}>
                    <span className={styles.metricLabel}>Predicted Stage</span>
                    <span className={styles.metricValue} style={{color:'var(--warning)'}}>{results.stage}</span>
                  </div>
                </div>

                <div className={styles.card} style={{borderColor: 'var(--accent-primary)', background: 'rgba(59, 130, 246, 0.05)'}}>
                  <h3 style={{fontSize:'1.125rem', marginBottom:'0.5rem', display:'flex', alignItems:'center', gap:'0.5rem'}}>
                    <Activity size={18} color="var(--accent-primary)" /> Anatomical Localization
                  </h3>
                  <p style={{fontSize:'1.125rem', fontWeight:'600'}}>{results.localization}</p>
                </div>

                <div className={styles.imagesGrid}>
                  <div className={styles.imageContainer}>
                    <div className={styles.imageLabel}>Fused AI Mask</div>
                    <div className={styles.placeholder}>
                      <Scan size={48} />
                      Segmentation Output
                    </div>
                  </div>
                  <div className={styles.imageContainer}>
                    <div className={styles.imageLabel}>Overlay Visualization</div>
                    <div className={styles.placeholder}>
                      <Layers size={48} />
                      Composite Detection
                    </div>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
}
