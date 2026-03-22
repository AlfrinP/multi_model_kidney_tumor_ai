"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { Activity, Brain, Layers, Microscope, Scan, ActivitySquare, ArrowRight } from "lucide-react";
import styles from "./page.module.css";

export default function Home() {
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
      },
    },
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.5 } },
  };

  return (
    <main className={styles.main}>
      <motion.section 
        className={styles.hero}
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
      >
        <div className={styles.badge}>
          <Activity size={16} />
          <span>v2.0 Multi-Modal PyTorch Pipeline</span>
        </div>
        
        <h1 className={styles.title}>
          Detect Kidney Tumors with <br/><span className={styles.titleHighlight}>AI Precision</span>
        </h1>
        
        <p className={styles.subtitle}>
          A state-of-the-art medical image processing platform. Upload coregistered CT and MRI scans to automatically detect, segment, and localize renal tumors using advanced U-Net fusion.
        </p>

        <div className={styles.ctaContainer}>
          <Link href="/test" className={styles.primaryButton}>
            Try the System <ArrowRight size={20} />
          </Link>
          <a href="#" className={styles.secondaryButton}>
            View Documentation
          </a>
        </div>
      </motion.section>

      <motion.div 
        className={styles.featuresGrid}
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        {[
          { icon: Layers, title: "Multi-Modal Fusion", desc: "Combines insights from both CT and MRI scan inputs for enhanced diagnostic confidence." },
          { icon: Scan, title: "Automated Segmentation", desc: "Identifies and precisely outlines the shape and bounds of the renal tumor in real-time." },
          { icon: ActivitySquare, title: "Volumetric Measurements", desc: "Automatically computes critical metrics including Tumor Diameter and Volume (cm³)." },
          { icon: Brain, title: "Stage Prediction", desc: "Predicts early versus advanced tumor stages via the TNM clinical categorization system." },
          { icon: Microscope, title: "Anatomical Localization", desc: "Classifies real-time spatial orientation determining Left/Right Kidney and Cortex Regions." },
          { icon: Activity, title: "Instant Inference", desc: "Optimized PyTorch inference pipeline delivers deep learning results in milliseconds." }
        ].map((feature, i) => (
          <motion.div key={i} className={styles.featureCard} variants={itemVariants}>
            <div className={styles.iconWrapper}>
              <feature.icon size={24} />
            </div>
            <h3 className={styles.featureTitle}>{feature.title}</h3>
            <p className={styles.featureDescription}>{feature.desc}</p>
          </motion.div>
        ))}
      </motion.div>

      <motion.section 
        className={styles.architectureSection}
        initial={{ opacity: 0 }}
        whileInView={{ opacity: 1 }}
        viewport={{ once: true, margin: "-100px" }}
        transition={{ duration: 0.8 }}
      >
        <h2 className={styles.architectureTitle}>How It Works</h2>
        <p className={styles.subtitle}>Deterministic fusion of two independent neural networks.</p>
        
        <div className={styles.architectureVisual}>
          <div className={styles.scanBox}>
            <h4 style={{marginBottom:'0.5rem', fontWeight:600, fontSize:'1.25rem'}}>CT Scan (UNet_CT)</h4>
            <p style={{fontSize:'0.9rem', color:'var(--text-muted)', lineHeight:1.5}}>High-res tissue anomaly detection</p>
          </div>
          <Activity className={styles.pulseIcon} size={32} />
          <div className={styles.scanBox} style={{borderColor: 'var(--accent-primary)', boxShadow: '0 0 30px rgba(59, 130, 246, 0.15)', transform: 'scale(1.05)'}}>
            <h4 style={{marginBottom:'0.5rem', fontWeight:600, fontSize:'1.25rem'}}>U-Net Fused Inference</h4>
            <p style={{fontSize:'0.9rem', color:'var(--text-muted)', lineHeight:1.5}}>Spatial & Gradient Features Combined</p>
          </div>
          <Activity className={styles.pulseIcon} size={32} />
          <div className={styles.scanBox}>
            <h4 style={{marginBottom:'0.5rem', fontWeight:600, fontSize:'1.25rem'}}>MRI Scan (UNet_MRI)</h4>
            <p style={{fontSize:'0.9rem', color:'var(--text-muted)', lineHeight:1.5}}>Deep structural contrast mapping</p>
          </div>
        </div>
      </motion.section>
    </main>
  );
}
